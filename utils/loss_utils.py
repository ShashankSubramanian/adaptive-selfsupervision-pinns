"""
  loss functions
"""
import torch
import torch.nn as nn
import logging
import numpy as np
from torch.autograd import Variable
import time
from utils.misc_utils import gaussian, poisson_source, get_diff_tensor, diffusion_coef, show, get_velocity
import matplotlib.pyplot as plt
#from functorch import vmap

class LossPinns():
    """ helper class to handles different PINNs losses """
    def __init__(self, params, model):
        self.params = params
        self.model = model
        self.der = None
        self.pde_residual = None
        self.loss_der = None
        self.source = None
        self.diff_coef = None
        self.temp_field = None # any extra field to visualize etc
        self.space_systems = ["poisson","poisadv"]
        if self.params.enable_obs and self.params.system not in self.space_systems: # observational ics; some systems dont have ics
            self.ics = [t*params.obs_dt for t in range(1, params.obs_nt+1)]

    def derivative(self, u, x):
        return torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]

    def ic(self, inputs, pred, target):
        if self.params.system not in self.space_systems:
            ic_indices = torch.where((target[:,1] == 5))[0]
#            ic_indices = torch.where((inputs[:,1] == 0.0))[0] # ic input
#            if self.params.enable_obs: # observational points are included
#                for ic in self.ics:
#                    obs = torch.where(torch.abs(inputs[:,1] - ic) < 1E-5)[0]
#                    ic_indices = torch.cat((ic_indices, obs), dim=0)
        else:
            if not self.params.enable_obs:
                ic_indices = [] # no ics
                return 0
            else: # repurpose the ic loss for obs
                ic_indices = torch.where((target[:,1] == 5))[0]

        pred_ic = pred[ic_indices,:]
        target_ic = target[ic_indices,0:1]

        if self.params.loss_style == 'mean':
            loss = torch.mean((target_ic - pred_ic)**2)
        elif self.params.loss_style == 'sum':
            loss = torch.sum((target_ic - pred_ic)**2)
        return loss

    def bc(self, inputs, pred, targets):
        if self.params.bc_type == "dirichlet":
            bc_indices = torch.where((targets[:,1]==0)|(targets[:,1]==1)|(targets[:,1]==2)|(targets[:,1]==3))[0]
            pred_bc = pred[bc_indices,:]
            target_bc = targets[bc_indices,0:1]
            if self.params.loss_style == 'mean':
                loss = torch.mean((pred_bc - target_bc)**2)
            elif self.params.loss_style == 'sum':
                loss = torch.sum((pred_bc - target_bc)**2)
#            bc_indices_x = torch.where((targets[:,1] == 0) | (targets[:,1] == 1))[0]
#            bc_indices_y = torch.where((targets[:,1] == 2) | (targets[:,1] == 3))[0]
#            pred_bc_x = pred[bc_indices_x,:]
#            target_bc_x = targets[bc_indices_x,0:1]
#            pred_bc_y = pred[bc_indices_y,:]
#            target_bc_y = targets[bc_indices_y,0:1]
#            if self.params.loss_style == 'mean':
#                loss = torch.mean((pred_bc_x - target_bc_x)**2)
#                if target_bc_y.shape[0] > 0:
#                    loss += torch.mean((pred_bc_y - target_bc_y)**2)
#            elif self.params.loss_style == 'sum':
#                loss = torch.sum((pred_bc_x - target_bc_x)**2)
#                if target_bc_y.shape[0] > 0:
#                    loss += torch.sum((pred_bc_y - target_bc_y)**2)
        else:
            # get the boundary indicies
            if self.params.system not in self.space_systems:
                b_indices = torch.where((inputs[:,0] == 0.0))[0] # boundary input
                b_indices_other_side = torch.where((inputs[:,0] == self.params.L))[0] # boundary input on top
            else:
                b_indices = torch.where((inputs[:,1] == 0.0))[0] # boundary input
                left = torch.where((inputs[:,0] == 0.0))[0] # left  bc
                b_indices = torch.cat((b_indices, left), dim=0) # this includes bottom and left indices
                b_indices_other_side = torch.where((inputs[:,1] == self.params.Ly))[0] # top
                right = torch.where((inputs[:,0] == self.params.Lx))[0] # right  bc
                b_indices_other_side = torch.cat((b_indices_other_side, right), dim=0) # top and right

            pred_lb = pred[b_indices,:]
            pred_ub = pred[b_indices_other_side,:]
            if self.params.loss_style == 'mean':
                loss = torch.mean((pred_lb - pred_ub)**2)
            elif self.params.loss_style == 'sum':
                loss = torch.sum((pred_lb - pred_ub)**2)

            if self.params.system in ["rd", "poisson", "poisadv"]: # bc on ders reqd because  second ders
                # der already computed for pde loss
                der_lb = self.der[b_indices,0]
                der_ub = self.der[b_indices_other_side,0]
                if self.params.loss_style == 'mean':
                    loss += torch.mean((der_lb - der_ub)**2)
                elif self.params.loss_style == 'sum':
                    loss += torch.sum((der_lb - der_ub)**2)

        return loss

    def pde(self, inputs, pred, targets, compute_loss_derivative=False):
        # find the interior points
#        if self.params.system != "poisson":
#            f_indices = torch.where(((inputs[:,0] != 0.0) & (inputs[:,0] != self.params.L)) & (inputs[:,1] != 0.0))[0] # interior
#        else:
        f_indices = torch.where((targets[:,1] == 4) | (targets[:,1] == -1))[0] # -1 is testing points
#            f_indices = torch.where(((inputs[:,0] != 0.0) & (inputs[:,0] != self.params.Lx)) & (inputs[:,1] != 0.0) & (inputs[:,1] != self.params.Ly))[0] # interior

        self.der = self.derivative(pred, inputs)
        der = self.der

        if self.params.system == "advection":
            ux = der[f_indices,0]
            ut = der[f_indices,1]
            g = self.params.source
            f_pred = ut + self.params.beta*ux - g
        elif self.params.system == "rd":
            u = pred[f_indices].flatten()
            ux = der[f_indices,0]
            ut = der[f_indices,1]
            lap = self.derivative(self.der[:,0], inputs) # derivative of du/dx
            uxx = lap[f_indices,0]
            f_pred = ut - self.params.rho*u*(1-u) - self.params.kappa*uxx
        elif self.params.system == "poisson":
#            ux = der[f_indices,0]
#            uy = der[f_indices,1]
            #self.source = poisson_source(inputs[:,0], inputs[:,1], self.params.force_mean, self.params.force_std, self.params, torch_tensor=True)
            self.source = gaussian(inputs[:,0], inputs[:,1], self.params.force_mean, self.params.force_std, self.params, torch_tensor=True)
            self.diff_coef = diffusion_coef(inputs[:,0], inputs[:,1], freq=self.params.diff_coef_freq, scale=self.params.diff_coef_scale, torch_tensor=True)

            # probe the source and diff tensor
            g = self.source[f_indices]

#            lap_x = self.derivative(der[:,0], inputs) # derivative of du/dx
#            lap_y = self.derivative(der[:,1], inputs) # derivative of du/dy

            # Kgrad
            K = get_diff_tensor(diff_type=self.params.diff_tensor_type)
            Kux = K['k11']*der[:,0] + K['k12']*der[:,1]
            Kuy = K['k22']*der[:,1] + K['k12']*der[:,0]
            # heterogeneous
            self.temp_field = self.diff_coef 
            Kux *= self.diff_coef
            Kuy *= self.diff_coef
            # div = d/dx(Kux) + d/dy(Kuy)
            fac = self.derivative(Kux, inputs)[f_indices,0] + self.derivative(Kuy, inputs)[f_indices,1]

#            laplacian = lap_x[:,0] + lap_y[:,1]
#            uxx = lap_x[f_indices,0] # d/dx of ux
#            uyy = lap_y[f_indices,1] # d/dy of uy
#            uxy = lap_y[f_indices,0] # d/dx of uy
#            uyx = lap_x[f_indices,1] # d/dy of ux
#           #f_pred = self.params.diff * (uxx + uyy) + g
#            fac = K['k11']*uxx + K['k22']*uyy + K['k12']*uxy + K['k12']*uyx

            f_pred = fac + g
        elif self.params.system == "poisadv":
            self.source = gaussian(inputs[:,0], inputs[:,1], self.params.force_mean, self.params.force_std, self.params, torch_tensor=True)
            self.diff_coef = diffusion_coef(inputs[:,0], inputs[:,1], freq=self.params.diff_coef_freq, scale=self.params.diff_coef_scale, torch_tensor=True)

            # probe the source and diff tensor
            g = self.source[f_indices]
            # Kgrad
            K = get_diff_tensor(diff_type=self.params.diff_tensor_type)
            v = get_velocity(vel_type=self.params.vel_type)

            vgradu = v['v1']*der[:,0] + v['v2']*der[:,1]

            Kux = K['k11']*der[:,0] + K['k12']*der[:,1]
            Kuy = K['k22']*der[:,1] + K['k12']*der[:,0]
            # heterogeneous
            self.temp_field = vgradu 
            Kux *= self.diff_coef
            Kuy *= self.diff_coef
            # div = d/dx(Kux) + d/dy(Kuy)
            fac = self.derivative(Kux, inputs)[f_indices,0] + self.derivative(Kuy, inputs)[f_indices,1]
            # -v.\gradu add
#            vtem = vgradu[f_indices]
#            logging.info("norm of pdes = vgradu {}, div {}, src {}".format(torch.norm(vtem), torch.norm(fac), torch.norm(g)))
            fac -= vgradu[f_indices]

            f_pred = fac + g

        self.pde_residual = f_pred
        if self.params.loss_style == 'mean':
            loss = torch.mean(f_pred**2)
        elif self.params.loss_style == 'sum':
            loss = torch.sum(f_pred**2)

        if compute_loss_derivative:
            # compute dL/dx
            gradloss = self.derivative(loss, inputs)[f_indices,:]
            self.loss_der = torch.sum(torch.abs(gradloss), dim=1)
            #self.loss_der = torch.sum(gradloss**2, dim=1)

        return loss

class LossMSE():
    """ mse loss """
    def __init__(self, params, model):
        self.params = params
        self.model = model

    def ic(self, inputs, pred_ic, target_ic):
        pred_ic = pred_ic
        target_ic = target_ic[:,0:1]
#        img = target_ic.detach().cpu().numpy()
#        img = img[1024:]
#        img = img.reshape((255, 255))
#        fig, ax = plt.subplots(1, 1, figsize=(8.5,4))
#        show(img, ax, fig)
#        fig.savefig("./target.pdf", format="pdf", dpi=1200, bbox_inches="tight")
#        exit(1)
        if self.params.loss_style == 'mean':
            loss = torch.mean((target_ic - pred_ic)**2)
        elif self.params.loss_style == 'sum':
            loss = torch.sum((target_ic - pred_ic)**2)
        return loss

    def bc(self, inputs, pred, targets):
        return 0

    def pde(self, inputs, pred, targets):
        return 0


class PyHessianLoss(nn.Module):
    """ wrapper class for loss routines to interface with pyhessian which requires loss to have
        two inputs only """ 
    def __init__(self, lf, params, inputs):
        super().__init__()
        self.loss_func = lf
        self.params = params
        self.inputs = inputs

    def forward(self, u, targets):
        inputs = self.inputs
        loss_ic = self.loss_func.ic(inputs, u, targets)
        loss_pde = self.loss_func.pde(inputs, u, targets)
        loss_bc = self.loss_func.bc(inputs, u, targets)
        loss = loss_ic + loss_bc + self.params.reg * loss_pde
        return loss


# pinns version of BC
#    def bc_(self, inputs, pred):
#        # get the boundary indicies
#        b_indices = torch.where((inputs[:,0] == 0.0))[0] # boundary input
#        lb = inputs[b_indices,:]
#        ub = lb.clone()
#        ub[:,0] =  self.params.L
#        pred_lb = pred[b_indices,:]
#        pred_ub = self.model(ub)
#        if self.params.loss_style == 'mean':
#            loss = torch.mean((pred_lb - pred_ub)**2)
#        elif self.params.loss_style == 'sum':
#            loss = torch.sum((pred_lb - pred_ub)**2)
#        return loss
#

# sort the indices if shuffled
#        assert len(b_indices) > 0, "no lower boundary!"
#        assert len(b_indices_other_side) > 0, "no upper boundary!"
        # sort acc to time
#        t_lb = inputs[b_indices,1]
#        t_ub = inputs[b_indices_other_side,1]
#        b_indices = [i for _,i in sorted(zip(t_lb, b_indices))]
#        b_indices_other_side = [i for _,i in sorted(zip(t_ub, b_indices_other_side))]
