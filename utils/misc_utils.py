"""
  misc utils
"""

import numpy as np
import scipy as sc
import scipy.ndimage as nd
import torch
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import Normalize
from matplotlib import cm

def get_velocity(vel_type="iso"):
    if vel_type == "iso":
        v = {'v1': 1, "v2": 1}
    elif vel_type == "aniso":
        v = {'v1': 40, "v2": 10}
    else:
        v = None
    return v
        

def get_diff_tensor(diff_type="iso"):
    if diff_type == "iso":
        K = {'k11': 1, 'k22': 1, 'k12': 0} # diffusion tensor; isotropic, circular
    elif diff_type == "aniso":
        K = {'k11': 1, 'k22': 8, 'k12': 2} # aniso
    else:
        K = None
    return K

def fft_coef(n):
    ikx_pos = 1j * np.arange(0, n//2+1, 1)
    ikx_neg = 1j * np.arange(-n//2+1, 0, 1)
    ikx = np.concatenate((ikx_pos, ikx_neg))
    return ikx

def get_fft_coef(u):
    nx, ny = u.shape
    ikx = fft_coef(nx).reshape(1,nx)
    ikx = np.repeat(ikx, ny, axis=0)
    iky = fft_coef(ny).reshape(ny,1)
    iky = np.repeat(iky, nx, axis=1)
    return ikx, iky

def laplacian(u, params):
    ikx, iky = get_fft_coef(u)
    ikx2 = ikx**2
    iky2 = iky**2
    u_hat = np.fft.fft2(u)
    u_hat *= (ikx2+iky2) * (4.0 * np.pi**2)/(params.Lx*params.Ly)
    return np.real(np.fft.ifft2(u_hat))

def grad(u, params):
    ikx, iky = get_fft_coef(u)
    u_hat = np.fft.fft2(u) 
    ux = np.real(np.fft.ifft2(u_hat * ikx)) * (2.0 * np.pi)/params.Lx
    uy = np.real(np.fft.ifft2(u_hat * iky)) * (2.0 * np.pi)/params.Ly
    return ux, uy

def gradabs(u, params):
    ux, uy = grad(u, params)
    return np.sqrt(ux**2 + uy**2)

def div(ux, uy, params):
    ikx, iky = get_fft_coef(ux)
    ux_hat = np.fft.fft2(ux)
    uy_hat = np.fft.fft2(uy)
    u1 = np.real(np.fft.ifft2(ux_hat * ikx)) * (2.0 * np.pi)/params.Lx
    u2 = np.real(np.fft.ifft2(uy_hat * iky)) * (2.0 * np.pi)/params.Ly
    return (u1 + u2)

def diffusion_coef(x, y, freq, scale, torch_tensor=False):
    if not torch_tensor:
        return (1 + scale*np.sin(2*np.pi*freq*x) * np.sin(2*np.pi*freq*y))
    else:
        return (1 + scale*torch.sin(2*np.pi*freq*x) * torch.sin(2*np.pi*freq*y))

def gaussian(x, y, mean, std, params, torch_tensor=False):
    if params.num_gauss == 1:
        r = (x - mean)*(x - mean) + (y - mean)*(y - mean)
        R = 2.*std*std
        ratio = [r/R]
        c = [1./(2*np.pi*std*std)] # normalizing factor
#        c = [1]
    else:
        ratio = []
        c = []
        for m, s in zip(mean, std):
            mx, my = m[0], m[1]
            r = (x - mx)**2 + (y - my)**2
            R = 2.*s*s
            ratio.append(r/R)
            c.append(1./(2*np.pi*s*s)) # normalizing factor

    source = 0*ratio[0]
    if torch_tensor:
        # use torch tensors during loss computations
        for r, ci in zip(ratio, c):
            source += ci*torch.exp(-r)
        dc = params.dc
        source = source - dc
    else:
        # compute for data; need dc component here
        for r, ci in zip(ratio, c):
            source += ci*np.exp(-r)
        dc = np.mean(source.flatten())
        source = source - dc
        params.dc = dc # set the dc comp
        #params.max_source_val = np.max(source.flatten())

    return source

def poisson_source(x, y, mean, std, params, torch_tensor=False):
    r = (x - mean)*(x - mean) + (y - mean)*(y - mean)
    R = 2.*std*std
    ratio = r/R

    fac = torch.exp(-ratio) if torch_tensor else np.exp(-ratio)

    f = ((x-mean)/std**2)**2 * fac - (2/std**2)*fac + ((y-mean)/std**2)**2 * fac
    if torch.tensor:
        return -(f - params.dc)
    else:
        params.dc = np.mean(f.flatten())
        return -(f - params.dc)
    

def softmax(x):
    return np.exp(x)/sum(np.exp(x))

def normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))

def compute_grad_norm(p_list):
    grad_norm = 0
    for p in p_list:
        param_g_norm = p.grad.detach().data.norm(2)
        grad_norm += param_g_norm.item()**2
    grad_norm = grad_norm**0.5
    return grad_norm

def compute_err(output, target):
    err = output - target
    return np.linalg.norm(err[:])/np.linalg.norm(target[:])

def show(u, ax, fig, rescale=None):
    if u is not None:
        if rescale is None:
            h = ax.imshow(u.T, interpolation='nearest', cmap='rainbow',
                        origin='lower', aspect='auto')
        else:
            h = ax.imshow(u.T, interpolation='nearest', cmap='rainbow',
                        origin='lower', aspect='auto', vmin=rescale[0], vmax=rescale[1])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.10)
        cbar = fig.colorbar(h, cax=cax)
        cbar.ax.tick_params(labelsize=15)
#    ax.set_xlabel('y', fontweight='bold', size=15)
#    ax.set_ylabel('x', fontweight='bold', size=15)
        ax.tick_params(labelsize=15)

def show3d(f, ax, fig):
    (x, y, u) = f
    surf = ax.plot_surface(x, y, u, cmap=cm.coolwarm)
    #cbar = fig.colorbar(surf, cax=cax)
#    cbar.ax.tick_params(labelsize=15)
#    ax.set_xlabel('y', fontweight='bold', size=15)
#    ax.set_ylabel('x', fontweight='bold', size=15)
#    ax.tick_params(labelsize=15)
    ax.set_xlabel('epsilon1')
    ax.set_ylabel('epsilon2')

def vis_fields(fields, params, domain):
    pred, tar, col_points, res, ic_points, src, uxx = fields
    space_systems = ["poisson","poisadv"]
    fx = 17 if params.system not in space_systems else 13.5
    fy = 8
    num_cols = 2 if params.system not in space_systems else 3
    show_contours = True if params.system in space_systems else False
    x_g = domain.x_g
    y_g = domain.y_g
    temp = domain.temp

    scale = [params.nt/params.T, params.nx/params.L] if params.system not in space_systems else [params.ny/params.Ly, params.nx/params.Lx]

    if params.in_dim == 2: # 2d
#        fig, ax = plt.subplots(2, num_cols, figsize=(fx,fy))
        fig = plt.figure(figsize=(fx,fy))
        ax1 = fig.add_subplot(2,num_cols,1)
        show(pred, ax1, fig)
        ax1.set_title("pred")
        ax2 = fig.add_subplot(2,num_cols,2)
        show(tar, ax2, fig)
        ax2.set_title("target")
        if show_contours:
            ax1.contour(y_g*scale[0], x_g*scale[1], pred, 15)
            ax2.contour(y_g*scale[0], x_g*scale[1], tar, 15)
        if not params.all_col_points:
            ax1.scatter(col_points[:,1]*scale[0], col_points[:,0]*scale[1], c='k', s=1)
        ax1.scatter(ic_points[:,1]*scale[0], ic_points[:,0]*scale[1], c='r', s=1)

        ax = fig.add_subplot(2,num_cols,3)
        err = np.abs(pred-tar)
        show(err, ax, fig)
        ax.set_title("l1 error")
        if not params.all_col_points:
            ax.scatter(col_points[:,1]*scale[0], col_points[:,0]*scale[1], c='k', s=1)
        ax.scatter(ic_points[:,1]*scale[0], ic_points[:,0]*scale[1], c='r', s=1)
#        show(err, ax[1][0], fig, rescale=[0,0.04])
#        show(np.abs(res), ax[1][1], fig, rescale=[0,0.1])
        ax = fig.add_subplot(2,num_cols,4)
        absres = np.abs(res) if res is not None else None
        show(absres, ax, fig)
        ax.set_title("l1 pde-res ")
        if not params.all_col_points:
            ax.scatter(col_points[:,1]*scale[0], col_points[:,0]*scale[1], c='k', s=1)
        ax.scatter(ic_points[:,1]*scale[0], ic_points[:,0]*scale[1], c='r', s=1)

        if num_cols > 2:
            if params.vis_loss:
                ax5 = fig.add_subplot(2,num_cols,5, projection='3d')
                ax6 = fig.add_subplot(2,num_cols,6, projection='3d')
                ax5.set_title("loss before")
                show3d(src, ax5, fig)
                ax6.set_title("loss after")
                show3d(uxx, ax6, fig)
            else:
                ax5 = fig.add_subplot(2,num_cols,5)
                ax6 = fig.add_subplot(2,num_cols,6)
                ax5.set_title("temp1")
                show(src, ax5, fig)
                ax6.set_title("temp2")
                if temp is not None:
                    show(temp, ax6, fig)
                else:
                    show(uxx, ax6, fig)


#        # change col points just for vis
#        mask = np.abs(res).flatten()
#        mask /= np.sum(mask) # prob
#        extra_idx = np.random.choice(mask.shape[0], 500, replace=False, p=mask)
#        vis_points = domain.grid[extra_idx,:]


#        # scatter plot the points
#        for i in range(2):
#            for j in range(num_cols):
#                if params.system != "poisson":
#                else:
##                    if i != 1 or j != 1:
#                    ax[i][j].scatter(col_points[:,1]*params.ny/params.Ly, col_points[:,0]*params.nx/params.Lx, c='k', s=1, zorder=1)
#                    ax[i][j].scatter(ic_points[:,1]*params.ny/params.Ly, ic_points[:,0]*params.nx/params.Lx, c='r', s=1, zorder=1)
##                    else:
##                    ax[i][j].scatter(vis_points[:,1]*params.ny/params.Ly, vis_points[:,0]*params.nx/params.Lx, c='k', s=1, zorder=1)
##                    ax[i][j].scatter(ic_points[:,1]*params.ny/params.Ly, ic_points[:,0]*params.nx/params.Lx, c='r', s=1, zorder=1)
        fig.tight_layout()
    else:
        fig = None
    return fig

def perturb_model(model_orig, model_perb, direction1, direction2, alpha1, alpha2):
    """Perturb in two directions."""
    changes = [alpha1*d1 + alpha2*d2 for (d1, d2) in zip(direction1, direction2)]

    for m_orig, m_perb, d in zip(model_orig.parameters(), model_perb.parameters(), changes):
        m_perb.data = m_orig.data + d
    
    return model_perb

def set_activation(activation):
    if activation == 'identity':
        return nn.Identity()
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'relu':
        return  nn.ReLU()
    elif activation == 'gelu':
        return nn.GELU()
    else:
        print("WARNING: invalid activation function!")
        return -1
