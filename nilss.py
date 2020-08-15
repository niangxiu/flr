# some twiked version:
# minimizing integrated L2 norm of Ju v + Jtilde eta
from __future__ import division
import numpy as np
from numpy import newaxis
import itertools
from pdb import set_trace
from scipy.linalg import block_diag
from M3u2 import *


def primal(u0):
    # ini_step is the total step number of u0
    # return quantities not related to fu: u, f, J, Ju
    u, Ju = np.zeros([2, nstep+1, nc])
    J = np.zeros([nstep+1])
    u[0] = u0
    for i in range(nstep):
        u[i+1], J[i], Ju[i] = fJJu(u[i])
    _, J[nstep], Ju[nstep] = fJJu(u[nstep])
    return u, J, Ju


def tangent(u, w0, vstar0):
    # return quantities related to fu: w, vstar
    w = np.empty([nstep+1, nc, nus])
    vstar = np.empty([nstep+1, nc])
    w[0] = w0
    vstar[0] = vstar0
    for i in range(nstep):
        fu, fs = fufs(u[i])
        w[i+1] = fu @ w[i]
        vstar[i+1] = fu @ vstar[i] + fs
    return w, vstar


def preprocess():
    # change alpha, betaJ, Javg, and maybe dJs(integration of Js, not in this case)
    global Javg
    Jus = np.empty([nseg_ps, nstep+1, nc])
    Js = np.empty([nseg_ps, nstep+1]) # the array for J, not dJ/ds
    np.random.seed()
    u0 = np.random.rand(nc)
    for i in range(nseg_ps):
        u, Js[i], Jus[i] = primal(u0)
        u0 = u[-1]
    Javg = Js[nseg_dis:,:-1].mean()
    return u0


def inner_products(u0, w0, vstar0):
    # return inner products on one segment, and xi's at the end of segments
    # no need to return delta xi's, they are used only inside this function
    u, J, Ju = primal(u0)
    w, vstar = tangent(u, w0, vstar0)
    weight = np.ones(nstep+1)
    weight[0] = weight[-1] = 0.5

    dwJu = (w * Ju[:,:,newaxis]).sum((0,1))
    dvstarJu = (vstar * Ju).sum()
    C = (w[:,:,:,newaxis] * w[:,:,newaxis,:] * weight[:,newaxis,newaxis,newaxis]).sum((0,1))
    Cinv = np.linalg.inv(C)
    dwvstar = (w* vstar[:,:,newaxis] * weight[:,newaxis,newaxis]).sum((0,1))

    return Cinv, dwvstar, dwJu, dvstarJu, u[-1], w[-1], vstar[-1], u, w, vstar, Ju


def nilss_k(Cinvs, ds, Rs, bs):
    # solve the nilss problem on k segments.
    kk = Cinvs.shape[0]
    Cinv = block_diag(*Cinvs)
    d = np.ravel(ds) 
    B = np.eye((kk-1)*nus, kk*nus, k=nus)
    B[:, :-nus] -= block_diag(*Rs)
    b = np.ravel(bs)
    
    lbd = np.linalg.solve(-B @ Cinv @ B.T, B @ Cinv @ d + b)
    a = -Cinv @ (B.T @ lbd + d)
    a = a.reshape([kk, nus])
    return a


def Q0q0(u0):
    # generate initial conditions
    w0 = np.random.rand(nc, nus)
    Q0, _ = np.linalg.qr(w0, 'reduced')
    q0 = np.zeros(nc)
    return Q0, q0


def renormalize(W, vstar):
    # take W, vstar at the end of segment k
    Q, R = np.linalg.qr(W, 'reduced')
    b = Q.T @ vstar
    q = vstar - Q @ b
    return Q, R, q, b


def getLEs(Rs):
    I = np.arange(Rs.shape[-1])
    _ = np.log2(np.abs(Rs[1:-1,I,I]))
    LEs = _.mean(axis=0) / nstep
    LEs = 2**LEs
    return LEs


def nilss(nseg):
    # the overall nilss algorithm
    Cinvs = np.empty([nseg, nus, nus])
    dwvstars, dwJus, aa = np.empty([3, nseg, nus])
    dvstarJus = np.empty([nseg])
    Rs = np.empty([nseg+1, nus, nus]) # R[0] is at t0, R[K] at T, but both not used
    bs = np.empty([nseg+1, nus])
    us, vstars, Jus = np.empty([3, nseg, nstep+1, nc]) # only for debug 
    ws = np.empty([nseg, nstep+1, nc, nus]) # only for debug and illustration

    u0 = preprocess()
    Q, q = Q0q0(u0)
    for k in range(nseg):
        Cinvs[k],dwvstars[k],dwJus[k], dvstarJus[k], u0, Wend, vstarend,\
                us[k], ws[k], vstars[k], Jus[k] \
                = inner_products(u0, Q, q)
        Q, Rs[k+1], q, bs[k+1] = renormalize(Wend, vstarend)

    LEs = getLEs(Rs)
    aa = nilss_k(Cinvs, dwvstars, Rs[1:-1], bs[1:-1])
    dJds = ((dwJus * aa).sum() + dvstarJus.sum()) / (nseg * nstep)
    v = vstars + (ws*aa[:,newaxis,newaxis,:]).sum(-1)

    return Javg, dJds, us, v, (Jus*v).sum(-1), LEs
