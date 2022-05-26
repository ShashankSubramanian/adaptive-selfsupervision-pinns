""" 
    train PINNs 
"""
import os, sys, time
import copy
import numpy as np
import argparse
import random
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import wandb
import matplotlib.pyplot as plt
from datetime import datetime
import logging
from utils import logging_utils
from pyhessian import hessian
logging_utils.config_logger()
from utils.YParams import YParams
from utils.domains import DomainXT, DomainXY
from utils.data_utils import get_data_loader, resample_data_loader
from utils.optimizer_utils import EarlyStopping, set_scheduler, set_optimizer
from utils.loss_utils import LossPinns, LossMSE, PyHessianLoss
from utils.misc_utils import compute_err, vis_fields, compute_grad_norm, perturb_model, gradabs

# models
import models.ffn

def set_seed(params, world_size):
    seed = params.seed
    if seed is None:
        seed = np.random.randint(10000)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if world_size > 0:
        torch.cuda.manual_seed_all(seed)

def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000

class Trainer():
    """ trainer class """
    def __init__(self, params, args):
        self.sweep_id = args.sweep_id
        self.root_dir = args.root_dir
        self.config = args.config 
        self.run_num = args.run_num
        params.log()
        self.log_to_screen = params.log_to_screen
        self.log_to_wandb = params.log_to_wandb
        params['name'] = args.config + '_' + args.run_num
        params['group'] = 'pinns_' + args.config
        set_seed(params, 1)
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.params = params
        self.params.device = self.device


    def build_and_launch(self):
        # init wandb
        if self.sweep_id:
            exp_dir = os.path.join(*[self.root_dir, 'sweeps', self.sweep_id, self.config])
        else:
            exp_dir = os.path.join(*[self.root_dir, 'expts', self.config, self.run_num])

        if not os.path.isdir(exp_dir):
            os.makedirs(exp_dir)
            os.makedirs(os.path.join(exp_dir, 'checkpoints/'))
            os.makedirs(os.path.join(exp_dir, 'wandb/'))

        self.params['experiment_dir'] = os.path.abspath(exp_dir)
        self.params['checkpoint_path'] = os.path.join(exp_dir, 'checkpoints/ckpt.tar')
        self.params['resuming'] = True if os.path.isfile(self.params.checkpoint_path) else False
        if self.log_to_wandb:
            if self.sweep_id:
                wandb.init(dir=os.path.join(exp_dir, "wandb"))
                hpo_config = wandb.config
                self.params.update_params(hpo_config)
                logging.info('HPO sweep %s, trial params:'%self.sweep_id)
                logging.info(self.params.log())
            else:
                wandb.init(dir=os.path.join(exp_dir, "wandb"),
                            config=self.params.params, name=self.params.name, group=self.params.group, project=self.params.project, 
                            entity=self.params.entity, resume=self.params.resuming)


        set_seed(self.params, 1)
        if params.system not in ["poisson", "poisadv"]:
            self.domain = DomainXT(self.params)
        else:
            self.domain = DomainXY(self.params)

        if self.params.batch_size == 'none':
            self.params.batch_size = self.domain.n_samples # full batch
        self.params['global_batch_size'] = self.params.batch_size
        self.params['local_batch_size'] = self.params.batch_size # makes no difference here (for ddp later)

        self.data_loader, self.data_loader_test, self.data_loader_val, self.domain = get_data_loader(self.params, False, self.domain)
        self.col_points = self.domain.col
        if params.system not in ["poisson","poisadv"]:
            self.ic_points = self.domain.ic
#            self.fields_to_plot = [np.zeros((self.params.nt, self.params.nx+1)).flatten() for _ in range(1)]
        else:
            self.bc_points = self.domain.bc_x
            self.bc_points = np.concatenate((self.bc_points, self.domain.bc_y), axis=0)
#            self.fields_to_plot = [np.zeros((self.params.ny+1, self.params.nx+1)).flatten() for _ in range(3)]
            self.ic_points = self.domain.obs if self.params.enable_obs else self.bc_points

        if self.params.vis_loss:
            self.top_eigenvector = None
            self.top_eigenvalues = None

        if self.params.track_eigv:
            self.curr_eigv = None

        self.prev_res = None
        self.resample = False
        self.plot_figs = False
        
        if self.params.model == 'ffn':
            self.model = models.ffn.ffn_pinns(self.params).to(self.device)
        else:
            logging.warning("model architecture invalid!")
            exit(1)

        self.optimizer = set_optimizer(self.params, self.model)
        if self.params.warmstart > 0:
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.params.adam_lr) # reset the optimizer to this

        self.scheduler = set_scheduler(self.params, self.optimizer)
        if self.params.loss_func == 'pinns':
            self.loss_func = LossPinns(self.params, self.model)
        elif self.params.loss_func == 'mse':
            self.loss_func = LossMSE(self.params, self.model)
        self.early_stopper = EarlyStopping()

        self.iters = 0
        self.startEpoch = 0

        if self.params.resuming:
            logging.info("Loading checkpoint %s"%self.params.checkpoint_path)
            self.restore_checkpoint(self.params.checkpoint_path)
        elif self.params.init_weights is not False:
            # load just the model weights
            logging.info("Loading model weights from %s"%self.params.init_weights)
            self.load_model(self.params.init_weights)

        self.epoch = self.startEpoch
        self.logs = {}
        self.train_loss = self.ic_loss = self.bc_loss = self.pde_loss = self.grad = self.rel_tr_err = 0.0
        n_params = count_parameters(self.model)
        if self.log_to_screen:
            logging.info(self.model)
            logging.info('number of model parameters: {}'.format(n_params))

        # launch training
        self.train()

    def train(self):
        if self.log_to_screen:
            logging.info("Starting training loop...")
        best_loss = np.inf

        warmstart = self.params.warmstart > 0
        warmstart_phase = True if warmstart else False
        save_resample_arr = True
        stall_resample = False # if optimizer stalls then resample points
        stalled_epoch = 0
        best_epoch = 0
        best_err = 1
        self.logs['best_epoch'] = best_epoch
        self.logs['best_test_err'] = best_err

        for epoch in range(self.startEpoch, self.params.max_epochs):
            self.epoch = epoch
            start = time.time()

            if warmstart:
                if self.epoch > self.params.warmstart and warmstart_phase:
                    # warmstart over
                    warmstart_phase = False
                    # reset optimizer
                    logging.info("Reseting optimizer after warmstart {} epochs".format(self.params.warmstart))
                    self.optimizer = set_optimizer(self.params, self.model)

            # train
            tr_time = self.train_one_epoch()
            val_time = self.val_one_epoch()
            self.logs['wt_norm'] = self.get_model_wt_norm(self.model)

            sample_freq = self.params.sample_freq
            cur_epoch = epoch - stalled_epoch # distance from stalled epoch (initially just epoch)
            self.resample = self.params.resample and ((cur_epoch+1)%sample_freq==0 or stall_resample)
            stall_resample = False # resampled at stall point, set it back to false
            plot_figs = self.params.plot_figs and (cur_epoch+1)%sample_freq==0
            self.plot_figs = plot_figs
            
            if self.params.test_every_epoch: # test every epoch to track testing error (only for analysis)
                test_time, fields = self.test_one_epoch()

                if self.params.resample:
                    cur = cur_epoch%self.params.resample_schedule_T
                    percentage = 0.5 * (1 + np.cos(cur/self.params.resample_schedule_T * np.pi))
                    self.logs['res_sched'] = percentage

                if self.resample:
                    # compute metrics to use for resampling procedure
                    res = fields[3] # pde residual or loss derivative
                    metric = np.abs(res)
                    if self.prev_res is not None:
                        # add momemtum to the res
                        metric += (self.params.res_momentum * self.prev_res)
                    self.prev_res = np.abs(res)

                    # plot the loss before resampling
                    if self.params.vis_loss:
                        fields[5] = self.plot_loss_landscape(compute_eig=True) 
                    self.data_loader = resample_data_loader(self.params, self.domain, metric, percentage)
                    self.col_points = self.domain.col
                    fields[2] = self.col_points
                    # plot the loss after resampling
                    if self.params.vis_loss:
                        fields[6] = self.plot_loss_landscape(compute_eig=False) 

            if self.params.track_eigv:
                if epoch == self.startEpoch or self.resample:
                    self.curr_eigv = self.compute_eigenvector()
                self.dot_with_eigv(self.curr_eigv)

            if self.params.scheduler == 'reducelr':
                self.scheduler.step(self.logs['train_loss'])
            elif self.params.scheduler == 'cosine':
                self.scheduler.step()

            if self.logs['val_loss'] <= best_loss:
                is_best_loss = True
                best_loss = self.logs['val_loss']
                best_err = self.logs['test_err']
            else:
                is_best_loss = False
            self.logs['best_val_loss'] = best_loss
            best_epoch = self.epoch if is_best_loss else best_epoch
            self.logs['best_epoch'] = best_epoch
            self.logs['best_test_err'] = best_err

            if self.params.save_checkpoint:
                #checkpoint at the end of every epoch
                if is_best_loss:
                    if save_resample_arr:
                        self.save_arrays(fields, tag="_best")
                    self.save_logs(tag="_best")
                if (self.params.resample and percentage<=1E-3):
                    if save_resample_arr:
                        self.save_arrays(fields, tag="_ep{}".format(self.epoch))
                    self.save_logs(tag="_ep{}".format(self.epoch))
                self.save_checkpoint(self.params.checkpoint_path, is_best=is_best_loss)

            if self.log_to_wandb:
                # log visualizations every epoch
                if plot_figs:
                    fig = vis_fields(fields, self.params, self.domain)
                    self.logs['vis'] = wandb.Image(fig)
                    plt.close(fig)
                self.logs['learning_rate'] = self.optimizer.param_groups[0]['lr']
                self.logs['time_per_epoch'] = tr_time
                wandb.log(self.logs, step=self.epoch+1)

            if self.log_to_screen:
                logging.info('Time taken for epoch {} is {} sec'.format(self.epoch+1, time.time()-start))
                logging.info('Loss (total = ic + bc + pde) {} = {} + {} + {}'.format(self.logs['train_loss'], self.logs['ic_loss'],
                self.logs['bc_loss'], self.logs['pde_loss']))

            #loss is increasing continuously early stopper
            if self.params.early_stop and self.params.optimizer == "lbfgs":
                if not warmstart_phase: # only early stop if not in a warmstart phase
                    self.early_stopper(self.logs['train_loss'])

            if self.resample:
                self.early_stopper.reset()

            if self.early_stopper.early_stop:
                if self.params.resample:
                    stall_resample = True
                    stalled_epoch = epoch
                    self.early_stopper.reset()
                else: 
                    #baseline
                    if not self.params.kill_opt:
                        self.data_loader = resample_data_loader(self.params, self.domain, None, 1, sample_uniformly=True)
                        self.col_points = self.domain.col
                        fields[2] = self.col_points
                        self.early_stopper.reset()

            if (epoch == self.params.max_epochs-1 or self.early_stopper.early_stop):
                logging.info('end of training or early stopper; running inference')
                test_time, fields = self.test_one_epoch()
                if self.log_to_wandb:
                    fig = vis_fields(fields, self.params, self.domain)
                    self.logs['vis'] = wandb.Image(fig)
                    plt.close(fig)
                    wandb.log(self.logs, step=self.epoch+1)
                else:
                    logging.info('testing error = {}'.format(self.logs['test_err']))

            if self.early_stopper.early_stop and (not self.params.resample) and (self.params.kill_opt): # baseline 
                break

        if self.log_to_wandb:
            wandb.finish()

    
    def get_model_wt_norm(self, model):
        n = 0
        for p in model.parameters():
            p_norm = p.data.norm(2)
            n += p_norm.item()**2
        n = n**0.5
        return n


    def train_one_epoch(self):
        tr_time = 0
        self.model.train()
        batch_size = self.params.local_batch_size # batch size per gpu
        self.logs['train_loss'] = self.logs['ic_loss'] = self.logs['bc_loss'] = self.logs['pde_loss'] = self.logs['grad'] = self.logs['rel_tr_err'] = 0.0
        for i, (inputs, targets) in enumerate(self.data_loader):
            self.iters += 1
            data_start = time.time()
#            inputs, targets = inputs.to(self.device), targets.to(self.device)
            inputs = Variable(inputs, requires_grad=True)
            targets = Variable(targets, requires_grad=True)
            tr_start = time.time()
            def closure():
                self.optimizer.zero_grad()
                u = self.model(inputs)
                # 3 different losses
                loss_ic = self.loss_func.ic(inputs, u, targets) # ic is simple mse
                loss_pde = self.loss_func.pde(inputs, u, targets)
                loss_bc = self.loss_func.bc(inputs, u, targets)
                loss = loss_ic + loss_bc + self.params.reg * loss_pde
                #loss = loss_ic + loss_bc + (1./self.params.max_source_val)*loss_pde
                rel_tr_err = torch.norm(u - targets[:,0:1])/torch.norm(targets[:,0:1]) # 2nd indx in target is marker
                # backprop
                if loss.requires_grad:
                    loss.backward()
                # losses and other logs
                # keep track of the loss (the last one is the correct one cause 
                # closure() is called multiple times during a step for linesearch)
                grad_norm = compute_grad_norm(self.model.parameters())
                self.train_loss = loss
                self.ic_loss = loss_ic
                self.bc_loss = loss_bc
                self.pde_loss = loss_pde
                self.grad = grad_norm
                self.rel_tr_err = rel_tr_err
                return loss

            if self.params.optimizer == "lbfgs":
                self.optimizer.step(closure)
            else:
                with torch.enable_grad():
                    loss = closure()
                self.optimizer.step()
    
            # add all the minibatch losses
            self.logs['train_loss'] += self.train_loss
            self.logs['ic_loss'] += self.ic_loss
            self.logs['bc_loss'] += self.bc_loss
            self.logs['pde_loss'] += self.pde_loss
            self.logs['grad'] += self.grad
            self.logs['rel_tr_err'] += self.rel_tr_err

            tr_time += time.time() - tr_start

        self.logs['train_loss'] /= len(self.data_loader)
        self.logs['ic_loss'] /= len(self.data_loader)
        self.logs['bc_loss'] /= len(self.data_loader)
        self.logs['pde_loss'] /= len(self.data_loader)
        self.logs['grad'] /= len(self.data_loader)
        self.logs['rel_tr_err'] /= len(self.data_loader)

        return tr_time

    def val_one_epoch(self):
        self.model.train() # need gradients
        val_start = time.time()

        for i, (inputs, targets) in enumerate(self.data_loader_val):
            val_inputs = Variable(inputs, requires_grad=True)
            val_targets = Variable(targets, requires_grad=True)
            outputs = self.model(val_inputs)
            # val losses
            loss_pde = self.loss_func.pde(val_inputs, outputs, val_targets)
            loss_bc = self.loss_func.bc(val_inputs, outputs, val_targets)
            loss = loss_bc + self.params.reg * loss_pde

        self.logs.update(
                {'val_bc_loss': loss_bc,
                 'val_pde_loss': loss_pde,
                 'val_loss': loss
                })

        val_time = time.time() - val_start

        return val_time

    def test_one_epoch(self):
        self.model.train() # need gradients
        test_start = time.time()

#        with torch.no_grad():
        pde_res = None
        loss_der = None
        for i, (inputs, targets) in enumerate(self.data_loader_test):
            test_inputs = Variable(inputs, requires_grad=True)
            test_targets = Variable(targets, requires_grad=True)
            outputs = self.model(test_inputs)
            if self.resample or self.plot_figs:
                # compute metrics for resampling here
                if self.params.use_grad_resample:
                    loss_pde = self.loss_func.pde(test_inputs, outputs, test_targets, compute_loss_derivative=True) # pde loss at all points
                else:
                    loss_pde = self.loss_func.pde(test_inputs, outputs, test_targets) # pde loss at all points
                loss_der = self.loss_func.loss_der
                pde_residual = self.loss_func.pde_residual
                source = self.loss_func.source
                temp_field = self.loss_func.temp_field

        ys = self.params.nt if self.params.system not in ["poisson","poisadv"] else self.params.ny + 1
        xs = self.params.nx + 1 
        outputs = outputs.reshape(ys, xs)

        if self.resample or self.plot_figs:
            pde_res = pde_residual.detach().cpu().numpy()
            #pde_res[self.domain.mask_interior] = pde_residual.detach().cpu().numpy()
            pde_res = pde_res.reshape(ys, xs)
            #pde_res = pde_res**2
            if loss_der is not None:
                loss_der = loss_der.detach().cpu().numpy()
                loss_der = loss_der.reshape(ys, xs)
        src = None
        uxx = None
        if params.system in ["poisson","poisadv"]:
            # remove the mean
            outputs -= torch.mean(outputs.view(-1))
            if self.resample or self.plot_figs:
                # plot additional fields
                src = source.detach().cpu().numpy()
                src = src.reshape(ys, xs)
                uxx = temp_field.detach().cpu().numpy()
                uxx = uxx.reshape(ys, xs)
            
#        grad_pred = gradabs(outputs.detach().cpu().numpy().reshape(ys, xs), self.params)

        if self.params.use_grad_resample:
            resample_metric = loss_der
#            if resample_metric is not None:
#                resample_metric += grad_pred
#            src = grad_pred
        else:
            resample_metric = pde_res

        sol = test_targets[:,0].reshape(ys, xs)
        fields = [outputs.detach().cpu().numpy(), sol.detach().cpu().numpy()
                  , self.col_points, resample_metric, self.ic_points
                  , src, uxx] 
        err = compute_err(fields[0], fields[1])

        self.logs.update(
                {'test_err': err
                })

        test_time = time.time() - test_start

        return test_time, fields

    def compute_eigenvector(self):
        for i, (inputs, targets) in enumerate(self.data_loader):
            # assume full batch
            inputs = Variable(inputs, requires_grad=True)
            targets = Variable(targets, requires_grad=True)
            break
        pyhess_loss_func = PyHessianLoss(self.loss_func, self.params, inputs)
        hessian_comp = hessian(self.model, pyhess_loss_func, data=(inputs, targets), cuda=True)
        _, ev = hessian_comp.eigenvalues()
        ev = torch.cat([e.view(-1) for e in ev[0]]) # flatten the eigenvector
        return ev

    def dot_with_eigv(self, ev):
        evcur = self.compute_eigenvector()
        dotpr = torch.dot(evcur, ev)
        logging.info("\ndot product of e_i wrt cur e = {}\n".format(dotpr))

    def plot_loss_landscape(self, compute_eig=True):
        # perturbation interp points
        n_interp = 21
        lams1 = np.linspace(-1, 1, n_interp).astype(np.float32)
        lams2 = np.linspace(-1, 1, n_interp).astype(np.float32)
        for i, (inputs, targets) in enumerate(self.data_loader):
            # assume full batch
            inputs = Variable(inputs, requires_grad=True)
            targets = Variable(targets, requires_grad=True)
            break

        pyhess_loss_func = PyHessianLoss(self.loss_func, self.params, inputs)
        if compute_eig:
            hessian_comp = hessian(self.model, pyhess_loss_func, data=(inputs, targets), cuda=True)
            self.top_eigenvalues, self.top_eigenvector = hessian_comp.eigenvalues(top_n=2) # top 2 eig
            if self.log_to_screen:
                logging.info("top eigenvalues = {},{}".format(self.top_eigenvalues[0], self.top_eigenvalues[1]))

        model_perb = copy.deepcopy(self.model) # deep copy model to perturb it
        loss_list = []
        for lam1 in lams1:
            for lam2 in lams2:
                model_perb = perturb_model(self.model, model_perb, self.top_eigenvector[0], self.top_eigenvector[1], lam1, lam2)
                loss_list.append(pyhess_loss_func(model_perb(inputs), targets).item())

        loss_list = np.array([loss_list])
        loss_list = loss_list.reshape(n_interp, n_interp)
        lx, ly = np.meshgrid(lams1, lams2)

        return (lx, ly, loss_list)

    def save_logs(self, tag=""):
        with open(os.path.join(self.params.experiment_dir, "logs"+tag+".txt"), "w") as f:
            f.write("epoch,{}\n".format(self.epoch))
            for k, v in self.logs.items():
                f.write("{},{}\n".format(k,v))

    def save_arrays(self, fields, tag=""):
        pred, tar, col_points, res, ic_points, src, uxx = fields
        np.save(os.path.join(self.params.experiment_dir, "pred"+tag+".npy"), pred)
        np.save(os.path.join(self.params.experiment_dir, "tar"+tag+".npy"), tar)
        np.save(os.path.join(self.params.experiment_dir, "col"+tag+".npy"), col_points)
        if src is not None:
            np.save(os.path.join(self.params.experiment_dir, "src"+tag+".npy"), src)
        if res is not None:
            np.save(os.path.join(self.params.experiment_dir, "metric"+tag+".npy"), np.abs(res))

    def save_checkpoint(self, checkpoint_path, is_best=False, model=None):
        if not model:
            model = self.model
        torch.save({'iters': self.iters, 'epoch': self.epoch, 'model_state': model.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict(), 'scheduler_state_dict': (self.scheduler.state_dict() if self.scheduler is not None else None)}, checkpoint_path)
        if is_best:
            torch.save({'iters': self.iters, 'epoch': self.epoch, 'model_state': model.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict(), 'scheduler_state_dict': (self.scheduler.state_dict() if  self.scheduler is not None else None)}, checkpoint_path.replace('.tar', '_best.tar'))

    def restore_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cuda:{}'.format(0))
        self.model.load_state_dict(checkpoint['model_state'])
        self.iters = checkpoint['iters']
        self.startEpoch = checkpoint['epoch'] + 1
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    def load_model(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cuda:{}'.format(0))
        self.model.load_state_dict(checkpoint['model_state'])

if __name__ == '__main__':
    # parsers
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_config", default='./config/pinns.yaml', type=str)
    parser.add_argument("--config", default='default', type=str)
    parser.add_argument("--root_dir", default='./', type=str, help='root dir to store results')
    parser.add_argument("--run_num", default='0', type=str, help='sub run config')
    parser.add_argument("--sweep_id", default=None, type=str, help='sweep config from ./configs/sweeps.yaml')
    args = parser.parse_args()
    params = YParams(os.path.abspath(args.yaml_config), args.config)
    trainer = Trainer(params, args)
    if args.sweep_id:
        logging.disable(logging.CRITICAL)
        wandb.agent(args.sweep_id, function=trainer.build_and_launch, count=1, project=trainer.params.project, entity=trainer.params.entity)
    else:
        trainer.build_and_launch()
