"""
  data loaders
"""
import re
import time
import os, sys
import logging
import glob
import torch
import random
import numpy as np
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.utils.data.distributed import DistributedSampler
import torchvision.transforms as transforms
from utils.pde_sols import compute_sol

def create_new_training_data(params, domain):
    # create new training data
    space_systems = ["poisson","poisadv"]
    if params.system not in space_systems:
        examples = np.concatenate([domain.ic, domain.bc, domain.col])
        sol_list = domain.sol # (sol, sol_ic, sol_bc, sol_col)
        target = (np.concatenate([sol_list[1], sol_list[2], sol_list[3]]))
    else:
        examples = np.concatenate([domain.bc_left, domain.bc_right, domain.bc_bot, domain.bc_top, domain.col])
        sol_list = domain.sol # (sol, sol_bc_x, sol_bc_y, sol_col, sol_obs)
        target = np.concatenate([sol_list[1], sol_list[2], sol_list[3], sol_list[4], sol_list[5]])
        if params.enable_obs:
            examples = np.concatenate([examples, domain.obs]) # observation points
            target = np.concatenate([target, sol_list[6]])
    return examples, target

def resample_data_loader(params, domain, mask, percentage, sample_uniformly=False):
    # resample 1-percentage col points in domain according to the mask
    domain.resample_col(mask, percentage, sample_uniformly) # mask is switched to prob inside resample_col
    params.batch_size = domain.n_samples # full batch; might have changed during resampling
    params['global_batch_size'] = params.batch_size
    params['local_batch_size'] = params.batch_size # makes no difference here (for ddp later)
    examples, target = create_new_training_data(params, domain)
    x_total = torch.tensor(examples, requires_grad=True).float().to(params.device)
    y_total = torch.tensor(target, requires_grad=True).float().to(params.device)
    train_dataset = TensorDataset(x_total, y_total)
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=params.batch_size, shuffle=False)
    logging.info("dataloader is resampled")
    return dataloader

def get_data_loader(params, distributed, domain):
    transform = torch.from_numpy
    space_systems = ["poisson","poisadv"]
    if params.system not in space_systems:
        dataset = SpaceTime(params, transform, domain)
    else:
        dataset = Space2D(params, transform, domain)

    sampler = DistributedSampler(dataset, shuffle=True) if distributed else None
    # overload the dataset with a simpler one
    # all tensors are moved to the gpu immediately because memory is not an issue
    x_total = torch.tensor(dataset.examples, requires_grad=True).float().to(params.device)
    y_total = torch.tensor(dataset.target, requires_grad=True).float().to(params.device)

    train_dataset = TensorDataset(x_total, y_total)
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=params.batch_size, shuffle=False)

    x_test = torch.tensor(dataset.examples_testing, requires_grad=True).float().to(params.device)
    y_test = torch.tensor(dataset.sol, requires_grad=True).float().to(params.device)
    test_dataset = TensorDataset(x_test, y_test)
    dataloader_test = torch.utils.data.DataLoader(test_dataset, batch_size=x_test.shape[0], shuffle=False)

    x_val = torch.tensor(dataset.examples_validation, requires_grad=True).float().to(params.device)
    y_val = torch.tensor(dataset.sol_val, requires_grad=True).float().to(params.device)
    val_dataset = TensorDataset(x_val, y_val)
    dataloader_val = torch.utils.data.DataLoader(val_dataset, batch_size=x_val.shape[0], shuffle=False)
#    dataloader_val = None
    
#    dataloader = DataLoader(dataset,
#                            batch_size=int(params.local_batch_size),
#                            num_workers=params.num_data_workers,
#                            shuffle=(sampler is None),
#                            sampler=sampler,
#                            drop_last=False,
#                            pin_memory=torch.cuda.is_available())
    return dataloader, dataloader_test, dataloader_val, dataset.domain

class SpaceTime(Dataset):
    def __init__(self, params, transform, domain):
        self.transform = transform
        self.params = params
        self.domain = domain
        # the training points are the ic, bc, and the collocation points
        self.examples = np.concatenate([self.domain.ic, self.domain.bc, self.domain.col])
        self.examples_testing = (self.domain.grid.astype('float32'))
        # ground truth everywhere
        self.sol_list = domain.sol # (sol, sol_ic, sol_bc, sol_col)
        self.target = (np.concatenate([self.sol_list[1], self.sol_list[2], self.sol_list[3]]))
        self.sol = (self.sol_list[0]) # full solution for testing

    def __len__(self):
        return self.domain.n_samples

    def __getitem__(self, idx):
        X = self.transform(self.examples[idx,:].astype('float32')) # x here is size (2,), so minibatch is (b,2)
        y = self.transform(self.target[idx,:].astype('float32'))  # full solution is the target (only used on ic in pinns)
        return X, y # (b,2), (b,1) data points


class Space2D(Dataset):
    def __init__(self, params, transform, domain):
        self.transform = transform
        self.params = params
        self.domain = domain
        self.examples = np.concatenate([self.domain.bc_left, self.domain.bc_right, self.domain.bc_bot, self.domain.bc_top, self.domain.col])
        self.examples_testing = (self.domain.grid.astype('float32'))
        self.examples_validation = (self.domain.val.astype('float32'))
        # ground truth everywhere
        self.sol_list = domain.sol # (sol, sol_bc_x, sol_bc_y, sol_col)
        self.target = (np.concatenate([self.sol_list[1], self.sol_list[2], self.sol_list[3], self.sol_list[4], self.sol_list[5]]))

        if self.params.enable_obs:
            self.examples = np.concatenate([self.examples, self.domain.obs]) # observation points
            self.target = np.concatenate([self.target, self.sol_list[6]])

        self.sol = (self.sol_list[0]) # full solution for testing
        self.sol_val = domain.sol_val # garbage in the interior

    def __len__(self):
        return self.domain.n_samples

    def __getitem__(self, idx):
        X = self.transform(self.examples[idx,:].astype('float32')) # x here is size (2,), so minibatch is (b,2)
        y = self.transform(self.target[idx,:].astype('float32'))  # full solution is the target (only used on ic in pinns)
        return X, y # (b,2), (b,2) data points


