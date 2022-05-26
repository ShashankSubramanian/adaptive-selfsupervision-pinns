'''
  domain classes
'''
import torch
import random
import numpy as np
from utils.pde_sols import compute_sol
from utils.misc_utils import normalize, softmax, show
import matplotlib.pyplot as plt

def fake_mask(nt, nx, n=100):
    mask = np.zeros((nt,nx))
    offset = 20
    mask[nt//4-offset:3*nt//4-offset, nx//4:3*nx//4] = 1
    nn = np.count_nonzero(mask)
    mask = mask/nn
    return mask

class DomainXT():
    """ 
        Creates a uniform grid of spatiotemporal points
    """
    def __init__(self, params):
        if params.L == '2pi':
            params.L = 2.0 * np.pi
        self.params = params
        dx = params.L / params.nx
        dt = params.T / params.nt
#        self.x = np.arange(0, params.L, dx)
        self.x = np.arange(0, params.L+dx, dx)
        self.t = np.arange(0, params.T, dt)
        x_g, t_g = np.meshgrid(self.x, self.t)
        self.x_g, self.y_g = x_g, t_g
        self.grid = np.column_stack((x_g.flatten(), t_g.flatten()))
        self.n_col = self.params.n_col
        mask_bc = np.logical_or(self.grid[:,0] == 0.0, self.grid[:,0] == params.L)
        self.temp = None

        if not params.enable_obs:
            mask_ic = np.logical_and((self.grid[:,1] == 0), ~mask_bc)
            mask_ic_t0 = mask_ic
        else:
            # ic is t=0 plus observational points
            ics = [t*params.obs_dt for t in range(1, params.obs_nt+1)]
            mask_ic = (self.grid[:,1] == 0) # t = 0
            mask_ic_t0 =  np.logical_and((self.grid[:,1] == 0), ~mask_bc) # only t=0
            for ic in ics:
                mask_ic = np.logical_or(mask_ic, np.abs(self.grid[:,1]-ic)<1E-5) # append the obs points (find nearest grid point) 
            # remove the bdries
            mask_ic = np.logical_and(mask_ic, ~mask_bc)

        mask_interior = np.logical_and(~mask_ic_t0, ~mask_bc) # all points not ic or bc
        self.ic = self.grid[mask_ic]
        self.bc = self.grid[mask_bc]
        self.interior = self.grid[mask_interior]
        all_col_points = self.params.all_col_points
        if all_col_points:
            self.n_col = self.interior.shape[0]
            print("Taking all {} col points".format(self.n_col))
        self.col, col_idx = self.uniform_sample(self.interior, self.n_col)

        extend = (params.mask is not False) # extend with this mask
        if extend:
            mask = np.load(params.mask)
            mask = mask.flatten()
            mask = mask[mask_interior] # get the interior mask
            mask = mask/(np.sum(mask)) # to probabilities
            self.col, col_idx = self.extend_col(self.interior, col_idx, mask)
            # perturb it
            self.col = self.perturb_col(self.col, h=(dx/2,dt/2))

            self.n_col = self.col.shape[0]
            print("Extended/resampled col points to {} points".format(self.n_col))

        self.n_ic = len(self.ic)
        self.n_bc = len(self.bc) 
        self.n_samples = (self.n_col + self.n_ic + self.n_bc)

        # ground truth everywhere
        self.mask_interior = mask_interior
        self.gt = compute_sol(self.x[:-1], self.t, self.params) # drop the last bc point
        self.gt = np.hstack((self.gt, self.gt[:,0:1])) # periodic, so last point sol is the same as zeroth point

        self.sol_ic = self.gt.flatten()[mask_ic]
        self.sol_ic = np.column_stack([self.sol_ic, 3*np.ones(self.sol_ic.shape)])

        self.sol_bc = self.gt.flatten()[mask_bc]
        self.sol_bc = np.column_stack([self.sol_bc, 0*np.ones(self.sol_bc.shape)])

        self.sol_interior = self.gt.flatten()[mask_interior]
        self.sol_col = self.sol_interior[col_idx] # sample the interior solution at the collocation points
        self.sol_col = np.column_stack([self.sol_col, 2*np.ones(self.sol_col.shape)])

        self.gt = self.gt.flatten()
        self.gt = np.column_stack([self.gt, -1*np.ones(self.gt.shape)]) # no marker for gt; -1 
        self.sol = [self.gt, self.sol_ic, self.sol_bc, self.sol_col]

    def uniform_sample(self, arr, n):
        idx = np.random.choice(arr.shape[0], n, replace=False)
        return arr[idx,:], idx

    def perturb_col(self, col, h=(np.pi/256,1/200)):
        eps_x = np.random.uniform(0, h[0], size=col.shape[0])
        col[:,0] += eps_x
        eps_t = np.random.uniform(0, h[1], size=col.shape[0])
        col[:,1] += eps_t
        return col

    def extend_col_all(self, arr, col_idx, mask):
        extra_idx = np.nonzero(mask)[0]
        col_idx = np.concatenate((col_idx, extra_idx)) # add the extra indices to original arr
        idx = np.unique(col_idx) # get the unique indices
        return arr[idx,:], idx

    def extend_col(self, arr, col_idx, mask, n):
        extra_idx = np.random.choice(arr.shape[0], n, replace=False, p=mask)
        if col_idx is not None:
            col_idx = np.concatenate((col_idx, extra_idx)) # add the extra indices to original arr
            idx = np.unique(col_idx) # get the unique indices
        else:
            idx = extra_idx
        return arr[idx,:], idx, arr[extra_idx,:]

    def resample_col(self, mask, percentage):
        """ resample col using percentage
            percentage points uniform and 1-percentage points resampled with mask
        """
        n_uni = int(self.params.n_col * percentage)
        n_res = self.params.n_col - n_uni

        # sample n_uni uniform points
        self.col, col_idx = self.uniform_sample(self.interior, n_uni)

        # get masks for resampling the rest
        dx = self.params.L / self.params.nx
        dt = self.params.T / self.params.nt
        mask = mask.flatten()
        mask = mask[self.mask_interior] # get the interior mask
        mask = mask/(np.sum(mask)) # to probabilities

        # extend points with resampled points
        self.col, col_idx, extra_col = self.extend_col(self.interior, col_idx, mask, n_res)
        self.col = self.perturb_col(self.col, h=(dx/2,dt/2))
        self.n_col = self.col.shape[0]
        self.n_samples = (self.n_col + self.n_ic + self.n_bc)
        # change the target to be computed at the resampled col points
        self.sol_col = self.sol_interior[col_idx] # sample the interior solution at the collocation points
        self.sol_col = np.column_stack([self.sol_col, 2*np.ones(self.sol_col.shape)])
        self.sol = [self.gt, self.sol_ic, self.sol_bc, self.sol_col]

class DomainXY():
    """ 
        Creates a uniform grid of 2D spatial points
    """
    def __init__(self, params):
        if params.Lx == '2pi':
            params.Lx = 2.0 * np.pi
        if params.Ly == '2pi':
            params.Ly = 2.0 * np.pi
        self.params = params
        dx = params.Lx / params.nx
        dy = params.Ly / params.ny
        self.dx = dx
        self.dy = dy
        self.x = np.arange(0, params.Lx+dx, dx)
        self.y = np.arange(0, params.Ly+dy, dy)
        x_g, y_g = np.meshgrid(self.x, self.y)
        self.x_g, self.y_g = x_g, y_g
        self.grid = np.column_stack((x_g.flatten(), y_g.flatten()))

        self.n_col = self.params.n_col
        mask_bc_left = (self.grid[:,0] == 0.0)
        mask_bc_right = (self.grid[:,0] == params.Lx)
        mask_bc_bot = (self.grid[:,1] == 0.0)
        mask_bc_bot = np.logical_and(mask_bc_bot, ~mask_bc_left)
        mask_bc_bot = np.logical_and(mask_bc_bot, ~mask_bc_right) # exclude the left and right points
        mask_bc_top = (self.grid[:,1] == params.Ly)
        mask_bc_top = np.logical_and(mask_bc_top, ~mask_bc_left)
        mask_bc_top = np.logical_and(mask_bc_top, ~mask_bc_right) # exclude the left and right points
        mask_bc_x = np.logical_or(mask_bc_left, mask_bc_right)
        mask_bc_y = np.logical_or(mask_bc_top, mask_bc_bot)
#        mask_bc_x = np.logical_or(self.grid[:,0] == 0.0, self.grid[:,0] == params.Lx)
#        mask_bc_y = np.logical_or(self.grid[:,1] == 0.0, self.grid[:,1] == params.Ly)
#        mask_bc_y = np.logical_and(mask_bc_y, ~mask_bc_x) # dont include x bc points
        mask_interior = np.logical_and(~mask_bc_x, ~mask_bc_y) # all points other points
        self.temp = None

        # 0 and 1 are left, right bcs, 2 is col points, 3 is obs/ic points if needed
        self.bc_x = self.grid[mask_bc_x]
        self.bc_y = self.grid[mask_bc_y]
        self.bc_left = self.grid[mask_bc_left]
        self.bc_right = self.grid[mask_bc_right]
        self.bc_top = self.grid[mask_bc_top]
        self.bc_bot  = self.grid[mask_bc_bot]
        self.interior = self.grid[mask_interior]
        all_col_points = self.params.all_col_points
        if all_col_points:
            self.n_col = self.interior.shape[0]
            print("Taking all {} col points".format(self.n_col))
            self.col = self.interior
            col_idx = list(range(0,self.n_col))
        else:
            self.col, col_idx = self.uniform_sample(self.interior, self.n_col)

        # add observation points
        if self.params.enable_obs:
            self.n_obs = self.params.n_obs
            self.obs, self.obs_idx = self.uniform_sample(self.interior, self.n_obs) 
        else:
            self.n_obs = 0

        self.n_bc_x = len(self.bc_x) 
        self.n_bc_y = len(self.bc_y) 
        self.n_samples = (self.n_col + self.n_bc_x + self.n_bc_y + self.n_obs)

        # ground truth everywhere
        self.mask_interior = mask_interior
        self.gt = compute_sol(self.x[:-1], self.y[:-1], self.params) # drop the last bc point
        self.gt = np.hstack((self.gt, self.gt[:,0:1])) # periodic, so last point sol is the same as zeroth point
        self.gt = np.vstack((self.gt, self.gt[0:1,:])) # periodic, so last point sol is the same as zeroth point

#        self.sol_bc_x = self.gt.flatten()[mask_bc_x]
#        self.sol_bc_y = self.gt.flatten()[mask_bc_y]
        self.sol_bc_left = self.gt.flatten()[mask_bc_left]
        self.sol_bc_right = self.gt.flatten()[mask_bc_right]
        self.sol_bc_top = self.gt.flatten()[mask_bc_top]
        self.sol_bc_bot = self.gt.flatten()[mask_bc_bot]
        # keep track of points with markers (bc, ic, obs, col)
        self.sol_bc_left = np.column_stack([self.sol_bc_left, 0*np.ones(self.sol_bc_left.shape)]) # append markers to point types
        self.sol_bc_right = np.column_stack([self.sol_bc_right, 1*np.ones(self.sol_bc_right.shape)]) # append markers to point types
        self.sol_bc_bot = np.column_stack([self.sol_bc_bot, 2*np.ones(self.sol_bc_bot.shape)]) # append markers to point types
        self.sol_bc_top = np.column_stack([self.sol_bc_top, 3*np.ones(self.sol_bc_top.shape)]) # append markers to point types
        
        self.sol_interior = self.gt.flatten()[mask_interior]
        self.sol_col = self.sol_interior[col_idx] # sample the interior solution at the collocation points
        self.sol_col = np.column_stack([self.sol_col, 4*np.ones(self.sol_col.shape)])

        self.gt = self.gt.flatten()
        self.gt = np.column_stack([self.gt, -1*np.ones(self.gt.shape)]) # no marker for gt

#        self.sol = [self.gt, self.sol_bc_x, self.sol_bc_y, self.sol_col]
        self.sol = [self.gt, self.sol_bc_left, self.sol_bc_right, self.sol_bc_bot, self.sol_bc_top, self.sol_col]
        if self.n_obs > 0:
            self.sol_obs = self.sol_interior[self.obs_idx]
            self.sol_obs = np.column_stack([self.sol_obs, 5*np.ones(self.sol_obs.shape)])
            self.sol.append(self.sol_obs)

        # validation points (fully random)
        self.val_interior, val_idx = self.uniform_random_sample_interior(self.params.n_val)
        self.sol_val_interior = self.gt.flatten()[val_idx] # garbage
        self.sol_val_interior = np.column_stack([self.sol_val_interior, -1*np.ones(self.sol_val_interior.shape)]) # no marker for val (pde loss computed everywhere)
        self.val_bc, self.sol_val_bc = self.random_sample_bc(self.params.nx//2) 
        self.val = np.concatenate([self.val_interior, self.val_bc])
        self.sol_val = np.concatenate([self.sol_val_interior, self.sol_val_bc])

    def uniform_random_sample_interior(self, n):
        x_vals = np.random.uniform(low=self.dx, high=self.params.Lx, size=(n,))
        y_vals = np.random.uniform(low=self.dy, high=self.params.Ly, size=(n,))
        # col_idxs are garbarge here, so trainng error is not measurable
        grid = np.column_stack([x_vals, y_vals])
        col_idx = np.array(list(range(0,n)))
        return grid, col_idx

    def random_sample_bc(self, n):
        ''' 
           randomly sample bc points from each boundary and interpolate the solution
        '''
        # bc_bot
        vals = np.sort(np.random.uniform(low=0, high=self.params.Lx+self.dx, size=(n,)))
        bc_bot_val = np.column_stack([vals, 0*vals]) # bottom
        sol_bc_bot_val = np.interp(bc_bot_val[:,0], self.bc_bot[:,0], self.sol_bc_bot[:,0])
        sol_bc_bot_val = np.column_stack([sol_bc_bot_val, 2*np.ones(sol_bc_bot_val.shape)]) # append markers to point types
        # bc_top
        vals = np.sort(np.random.uniform(low=0, high=self.params.Lx+self.dx, size=(n,)))
        bc_top_val = np.column_stack([vals, 0*vals+self.params.Ly]) # top
        sol_bc_top_val = np.interp(bc_top_val[:,0], self.bc_top[:,0], self.sol_bc_top[:,0])
        sol_bc_top_val = np.column_stack([sol_bc_top_val, 3*np.ones(sol_bc_top_val.shape)]) # append markers to point types
        # bc_left
        vals = np.sort(np.random.uniform(low=0, high=self.params.Ly+self.dy, size=(n,)))
        bc_left_val = np.column_stack([0*vals, vals]) # left
        sol_bc_left_val = np.interp(bc_left_val[:,1], self.bc_left[:,1], self.sol_bc_left[:,0])
        sol_bc_left_val = np.column_stack([sol_bc_left_val, 0*np.ones(sol_bc_left_val.shape)]) # append markers to point types
        # bc_right
        vals = np.sort(np.random.uniform(low=0, high=self.params.Ly+self.dy, size=(n,)))
        bc_right_val = np.column_stack([0*vals+self.params.Lx, vals]) # right
        sol_bc_right_val = np.interp(bc_right_val[:,1], self.bc_right[:,1], self.sol_bc_right[:,0])
        sol_bc_right_val = np.column_stack([sol_bc_right_val, 1*np.ones(sol_bc_right_val.shape)]) # append markers to point types

        val_bc = np.concatenate([bc_left_val, bc_right_val, bc_bot_val, bc_top_val])
        sol_val_bc = np.concatenate([sol_bc_left_val, sol_bc_right_val, sol_bc_bot_val, sol_bc_top_val])

        return val_bc, sol_val_bc


    def uniform_random_sample(self, n):
        x_vals = np.random.uniform(low=0, high=self.params.Lx+self.dx, size=(n,))
        y_vals = np.random.uniform(low=0, high=self.params.Ly+self.dy, size=(n,))
        # col_idxs are garbarge here, so trainng error is not measurable
        grid = np.column_stack([x_vals, y_vals])
        col_idx = np.array(list(range(0,n)))
        return grid, col_idx

    def uniform_sample(self, arr, n):
#        return self.uniform_random_sample_interior(n)
        idx = np.random.choice(arr.shape[0], n, replace=False)
        return arr[idx,:], idx

    def perturb_col(self, col, h=(np.pi/256,1/200)):
        eps_x = np.random.uniform(0, h[0], size=col.shape[0])
        col[:,0] += eps_x
        eps_t = np.random.uniform(0, h[1], size=col.shape[0])
        col[:,1] += eps_t
        return col

    def extend_col(self, arr, col_idx, mask, n):
        extra_idx = np.random.choice(arr.shape[0], n, replace=False, p=mask)
        if col_idx is not None:
            col_idx = np.concatenate((col_idx, extra_idx)) # add the extra indices to original arr
            idx = np.unique(col_idx) # get the unique indices
        else:
            idx = extra_idx
        return arr[idx,:], idx, arr[extra_idx,:]

    def resample_uniformly(self):
        self.col, col_idx = self.uniform_sample(self.interior, self.params.n_col)
        # change the target to have the new points
        self.sol_col = self.sol_interior[col_idx] # sample the interior solution at the collocation points
        self.sol_col = np.column_stack([self.sol_col, 4*np.ones(self.sol_col.shape)])

        self.sol = [self.gt, self.sol_bc_left, self.sol_bc_right, self.sol_bc_bot, self.sol_bc_top, self.sol_col]
        #self.sol = [self.gt, self.sol_bc_x, self.sol_bc_y, self.sol_col]
        if self.n_obs > 0:
            self.sol.append(self.sol_obs)
        return

    def resample_col(self, mask, percentage, sample_uniformly=False):
        """ resample col using percentage
            percentage points uniform and 1-percentage points resampled with mask
        """
        if sample_uniformly:
            # just resample points again
            return self.resample_uniformly()

        n_uni = int(self.params.n_col * percentage)
        n_res = self.params.n_col - n_uni

        # sample n_uni uniform points
        self.col, col_idx = self.uniform_sample(self.interior, n_uni)

        # get masks for resampling the rest
        dx = self.params.Lx / self.params.nx
        dy = self.params.Ly / self.params.ny
        mask = mask.flatten()
#        m2 = mask.copy()
        mask = mask[self.mask_interior] # get the interior mask
        mask += self.params.uniform_reg * np.sum(mask)
        mask = mask/(np.sum(mask)) # to probabilities
        self.temp = mask.reshape((self.params.nx-1, self.params.ny-1))

        self.col, col_idx, extra_col = self.extend_col(self.interior, col_idx, mask, n_res)
        self.col = self.perturb_col(self.col, h=(dx/2,dy/2))


        self.n_col = self.col.shape[0]
        self.n_samples = (self.n_col + self.n_bc_y + self.n_bc_x + self.n_obs)
        # change the target to be computed at the resampled col points
        self.sol_col = self.sol_interior[col_idx] # sample the interior solution at the collocation points
        self.sol_col = np.column_stack([self.sol_col, 4*np.ones(self.sol_col.shape)])

        self.sol = [self.gt, self.sol_bc_left, self.sol_bc_right, self.sol_bc_bot, self.sol_bc_top, self.sol_col]
#        self.sol = [self.gt, self.sol_bc_x, self.sol_bc_y, self.sol_col]
        if self.n_obs > 0:
            self.sol.append(self.sol_obs)

