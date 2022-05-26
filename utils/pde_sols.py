"""
   solutions to different PDE systems
"""
import time
import os
import numpy as np
import scipy as sc
import torch
import torch.fft
import logging
import matplotlib.pyplot as plt
from scipy.sparse.linalg import LinearOperator
from utils.misc_utils import gaussian, poisson_source, diffusion_coef, show, get_diff_tensor, fft_coef, laplacian, grad, div, get_velocity, gradabs

def create_ic(u_0: str):
    if u_0 == "sin(x)":
        return lambda x: np.sin(x)
    elif u_0 == "gauss":
        x0 = np.pi
        sigma = np.pi/16
        return lambda x: np.exp(-np.power((x - x0)/sigma, 2.)/2.)

def compute_sol(x, t, params):
    if params.system == "advection":
        return advection(x, t, params.ic, params.source, params.beta)
    elif params.system == "rd":
        return rd(x, t, params.ic, params.source, params.rho, params.kappa, dt=params.T/params.nt)
    elif params.system == "poisson":
        return poisson(x, t, params.force_std, params.force_mean, params) # h=(params.Lx/params.Nx,params.Ly/params.Ny))
    elif params.system == "poisadv":
        return poisadv(x, t, params.force_std, params.force_mean, params) # h=(params.Lx/params.Nx,params.Ly/params.Ny))

def poisson_(x, y, std, mean, params):
    x_g, y_g = np.meshgrid(x, y)
    u = gaussian(x_g, y_g, mean, std, params)
    source = poisson_source(x_g, y_g, mean, std, params)
    return (u - np.mean(u.flatten()))

def advection_op(u, v, nx, ny, params):
    u = u.reshape((nx, ny))    
    gradux, graduy = grad(u, params)
    return (v['v1']*gradux + v['v2']*graduy)

def diffusion_op(u, k, K, nx, ny, params):
    u = u.reshape((nx, ny))    
    gradux, graduy = grad(u, params)
    # diff tensor
    Kux = K['k11']*gradux + K['k12']*graduy
    Kuy = K['k22']*graduy + K['k12']*gradux
    # heterogeneous
    Kux *= k
    Kuy *= k
    return div(Kux, Kuy, params)
    

def numerical_poisson(x, y, std, mean, params):
    x_g, y_g = np.meshgrid(x, y)
    # gaussian source
    source = gaussian(x_g, y_g, mean, std, params) # this will set the mean source value
    m, n = x_g.shape
    save_sol_dir = os.path.join(*[params.experiment_dir, "../"])
    sol_file = os.path.join(save_sol_dir, "hetdiff.npy")
    K = get_diff_tensor(diff_type=params.diff_tensor_type)
    if not os.path.exists(sol_file):
        # diff coeff
        k = diffusion_coef(x_g, y_g, freq=params.diff_coef_freq, scale=params.diff_coef_scale)
        mv = lambda x : diffusion_op(x, k, K, m, n, params)
        A = LinearOperator((m*n,m*n), matvec=mv)
        b = -source.flatten()
        # cg solve
        t0 = time.time()
        u, info = sc.sparse.linalg.cg(A, b)
        logging.info("numerical pde solver converged in {} seconds with criterion {}".format(time.time()-t0, info))
        u = u - np.mean(u)
        np.save(sol_file, u)
    else:
        logging.info("loading solution file {}".format(sol_file))
        u = np.load(sol_file)
    return u.reshape((m,n))
    

def poisson(x, y, std, mean, params):
    if params.diff_coef_freq > 0:
        return numerical_poisson(x, y, std, mean, params)

    x_g, y_g = np.meshgrid(x, y)
    # gaussian source
    source = gaussian(x_g, y_g, mean, std, params)

    nx = x.shape[0]
    ny = y.shape[0]
    ikx = fft_coef(nx).reshape(1,nx)
    ikx = np.repeat(ikx, ny, axis=0)
    iky = fft_coef(ny).reshape(ny,1)
    iky = np.repeat(iky, nx, axis=1)
    ikx2 = ikx**2
    iky2 = iky**2

    f_hat = np.fft.fft2(source)
    K = get_diff_tensor(diff_type=params.diff_tensor_type)
    diff_factor = ikx2*K['k11'] + iky2*K['k22'] + 2*ikx*iky*K['k12']
    factor = np.where(diff_factor == 0, 0, -1/diff_factor)
    u_hat = factor * f_hat * params.Lx * params.Ly / (4.0 * np.pi**2)

    u = np.real(np.fft.ifft2(u_hat)) / params.diff_coef_scale

#    print("true sol max, min, mean = {},{},{}".format(np.max(u),np.min(u),np.mean(u)))

    u = u - np.mean(u.flatten())  # remove the dc component
    
#    res = laplacian(u)
#
#    print("res max,min,mean = {},{},{}".format(np.max(res), np.min(res), np.mean(res**2)))

#    fig, ax = plt.subplots(1, 2, figsize=(17,8))
#    show(source, ax[0], fig)
#    show(u, ax[1], fig)
#    ax[1].contour(y_g*256, x_g*256, u, 10)
#    fig.savefig("./pdesol.pdf", format="pdf", dpi=1200, bbox_inches="tight")
#    exit(1)

    return u

def poisadv(x, y, std, mean, params):
    """ a.\gradu + \lapu = f """
    x_g, y_g = np.meshgrid(x, y)
    # gaussian source
    source = gaussian(x_g, y_g, mean, std, params)

    nx = x.shape[0]
    ny = y.shape[0]
    ikx = fft_coef(nx).reshape(1,nx)
    ikx = np.repeat(ikx, ny, axis=0)
    iky = fft_coef(ny).reshape(ny,1)
    iky = np.repeat(iky, nx, axis=1)
    ikx2 = ikx**2
    iky2 = iky**2

    f_hat = np.fft.fft2(source)
    K = get_diff_tensor(diff_type=params.diff_tensor_type)
    v = get_velocity(vel_type=params.vel_type)

    diff_factor = ikx2*K['k11'] + iky2*K['k22'] + 2*ikx*iky*K['k12']
    diff_factor *= (4.0 * np.pi**2) / (params.Lx * params.Ly)
    adv_factor = v['v1']*ikx + v['v2']*iky
    adv_factor *= (2.0 * np.pi) / (params.Lx * params.Ly)
    factor = diff_factor - adv_factor

    factor = np.where(factor == 0, 0, -1/factor)
    u_hat = factor * f_hat

    u = np.real(np.fft.ifft2(u_hat)) / params.diff_coef_scale
    u = u - np.mean(u.flatten())  # remove the dc component

#    fig, ax = plt.subplots(1, 2, figsize=(17,8))
#    gabs = gradabs(u, params)
#    show(gabs, ax[0], fig)
#    show(u, ax[1], fig)
#    ax[1].contour(y_g*256, x_g*256, u, 10)
#    fig.savefig("./pdesol.pdf", format="pdf", dpi=1200, bbox_inches="tight")
#    exit(1)
#    m, n = x_g.shape
#    k = diffusion_coef(x_g, y_g, freq=params.diff_coef_freq, scale=params.diff_coef_scale)
#    diffu = diffusion_op(u, k, K, m, n, params)
#    advu = advection_op(u, v, m, n, params)
#    logging.info("true norm of pdes = vgradu {}, div {}, src {}".format(np.linalg.norm(advu), np.linalg.norm(diffu), np.linalg.norm(source)))
#    fig, ax = plt.subplots(2, 2, figsize=(17,17))
#    show(source, ax[0][0], fig)
#    show(u, ax[0][1], fig)
#    show(diffu, ax[1][0], fig)
#    show(advu, ax[1][1], fig)
#    ax[0][1].contour(y_g*256, x_g*256, u, 10)
#    ax[1][0].contour(y_g*256, x_g*256, diffu, 10)
#    ax[1][1].contour(y_g*256, x_g*256, advu, 10)
#    fig.savefig("./pdesol.pdf", format="pdf", dpi=1200, bbox_inches="tight")
#    exit(1)

    return u

def advection(x, t, u_0: str, source=0, beta=1):
    u_0 = create_ic(u_0)
    u_0 = u_0(x)
    x_g, t_g = np.meshgrid(x, t)
    n = x.shape[0]
    source = np.zeros(u_0.shape) + source
    ikx = fft_coef(n)
    u_hat_0 = np.fft.fft(u_0)
    nu_factor = np.exp(-1.0 * beta * ikx * t_g)
    a = u_hat_0 
    u_hat = a*nu_factor + np.fft.fft(source)*t_g
    u = np.real(np.fft.ifft(u_hat))
    return u

def reaction(u, rho, dt):
    """ du/dt = rho*u*(1-u)
    """
    factor_1 = u * np.exp(rho * dt)
    factor_2 = (1 - u)
    u = factor_1 / (factor_2 + factor_1)
    return u

def diffusion(u, kappa, dt, ikx2):
    """ du/dt = kappa*d2u/dx2
    """
    factor = np.exp(kappa * ikx2 * dt)
    u_hat = np.fft.fft(u)
    u_hat = u_hat * factor
    u = np.real(np.fft.ifft(u_hat))
    return u

def rd(x, t, u_0: str, source=0, rho=1, kappa=0.01, dt=1E-2):
    u_0 = create_ic(u_0)
    u_0 = u_0(x)
    n = x.shape[0]
    nt = t.shape[0]
    ikx = fft_coef(n)
    ikx2 = ikx*ikx
    u = np.zeros((nt, n))
    u[0,:] = u_0
    u_ = u_0 # sol start
    for i in range(nt-1):
        # dt is temporal discr; so sol has truncation err dt=1e-3
        u_ = reaction(u_, rho, dt)
        u_ = diffusion(u_, kappa, dt, ikx2)
        u[i+1,:] = u_
    return u
