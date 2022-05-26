import os, sys
import logging
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.mod_lbfgs import ModLBFGS

class EarlyStopping:
    """ https://github.com/Bjarten/early-stopping-pytorch """
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=10, delta=1E-7):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def reset(self):
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        logging.info('EarlyStopper reset')

    def __call__(self, val_loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            logging.info('EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            # reset counter if loss is reducing
            self.best_score = score
            self.counter = 0


def set_scheduler(args, optimizer):
    """ set the lr scheduler """
    if args.scheduler == 'reducelr':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=args.patience, verbose=True, min_lr=1e-3*1e-5, factor=0.2)
    elif args.scheduler == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs, eta_min=1E-6)
    else:
        scheduler = None
    return scheduler

def set_optimizer(args, net):
    """ set the optimizer """
    if args.optimizer == "adam":
        optimizer = optim.Adam(net.parameters(), lr=args.lr)
    elif args.optimizer == "sgd":
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)    
    elif args.optimizer == "lbfgs":
#        optimizer = optim.LBFGS(net.parameters(), lr=args.lr, max_iter=args.max_iter, max_eval=None,
#                            history_size=50, tolerance_grad=1e-7, tolerance_change=1e-7,
#                            line_search_fn="strong_wolfe")
        optimizer = ModLBFGS(net.parameters(), lr=args.lr, max_iter=args.max_iter, max_eval=None,
                            history_size=50, tolerance_grad=1e-7, tolerance_change=1e-7,
                            line_search_fn="strong_wolfe")
    return optimizer

