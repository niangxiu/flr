# some twiked version:
# minimizing integrated L2 norm of Ju v + Jtilde eta
from __future__ import division
import numpy as np
from numpy import newaxis
import itertools
from pdb import set_trace
from scipy.linalg import block_diag
from ds import *


def primal_raw(u0, n):
    # return quantities not related to fu: u, J, Ju
    u, Ju = np.zeros([2, n+1, nc])
    J = np.zeros([n+1])
    u[0] = u0
    for i in range(n):
        u[i+1], J[i], Ju[i] = fJJu(u[i])
    _, J[n], Ju[n] = fJJu(u[n])
    return u, J, Ju


def primal(u0, nseg, W):
    # compute psi and reshape the raw results. Note that the returned u[0,0] is not u0.
    u, Ju = np.nan * np.empty([2, nseg, nstep+1, nc]) # only for debug 
    psi = np.zeros([nseg, nstep+1])

    u_, J_, Ju_ = primal_raw(u0, nseg*nstep + 2*W)
    # u_, J_, Ju_ = primal_raw(u0, nseg*nstep + 10)
    Javg = J_.mean()
    J_ = J_ - Javg
    for k in range(nseg):
        u[k]= u_[k*nstep+W: k*nstep+W+nstep+1]
        Ju[k]= Ju_[k*nstep+W: k*nstep+W+nstep+1]
        for w in range(2*W+1):
            psi[k] += J_[k*nstep+w: k*nstep+w+nstep+1]
        # u[k]= u_[k*nstep+2: k*nstep+nstep+3]
        # Ju[k]= Ju_[k*nstep+2: k*nstep+nstep+3]
        # psi[k] = J_[k*nstep+1: k*nstep+nstep+2]
    return u, Ju, psi, Javg


def preprocess():
    # return the initial condition 
    np.random.seed()
    u0 = np.random.rand(nc)
    u, _, _ = primal_raw(u0, nseg_ps*nstep)
    return u[-1]


def tangent(u, w0, vstar0, psi, vtstar0):
    # return quantities related to fu: w, vstar
    w = np.nan * np.empty([nstep+1, nc, nus])
    vstar, vtstar = np.nan * np.empty([2, nstep+1, nc]) # vt is tilde v
    w[0] = w0
    vstar[0] = vstar0
    vtstar[0] = vtstar0
    for i in range(nstep):
        fu, fs = fufs(u[i])
        w[i+1] = fu @ w[i]
        vstar[i+1] = fu @ vstar[i] + fs
        vtstar[i+1] = fu @ vtstar[i] + fs * psi[i]
    return w, vstar, vtstar


def inner_products(u, Ju, w0, vstar0, psi, vtstar0):
    # return inner products on one segment
    w, vstar, vtstar = tangent(u, w0, vstar0, psi, vtstar0)
    weight = np.ones(nstep+1)
    weight[0] = weight[-1] = 0.5

    dwJu = (w * Ju[:,:,newaxis] * weight[:,newaxis,newaxis]).sum((0,1))
    dvstarJu = (vstar * Ju * weight[:,newaxis]).sum()
    dvtstarJu = (vtstar * Ju * weight[:,newaxis]).sum()
    C = (w[:,:,:,newaxis] * w[:,:,newaxis,:] * weight[:,newaxis,newaxis,newaxis]).sum((0,1))
    Cinv = np.linalg.inv(C)
    dwvstar = (w* vstar[:,:,newaxis] * weight[:,newaxis,newaxis]).sum((0,1))
    dwvtstar = (w* vtstar[:,:,newaxis] * weight[:,newaxis,newaxis]).sum((0,1))

    return Cinv, dwvstar, dwvtstar, dwJu, dvstarJu, dvtstarJu, w, vstar, vtstar


def Q0q0():
    # generate initial conditions
    w0 = np.random.rand(nc, nus)
    Q0, _ = np.linalg.qr(w0, 'reduced')
    q0 = np.zeros(nc)
    qt0 = np.zeros(nc)
    return Q0, q0, qt0


def renormalize(W, vstar, vtstar):
    Q, R = np.linalg.qr(W, 'reduced')
    b = Q.T @ vstar
    q = vstar - Q @ b
    bt = Q.T @ vtstar
    qt = vtstar - Q @ bt
    return Q, R, q, b, qt, bt


def getLEs(Rs):
    I = np.arange(Rs.shape[-1])
    _ = np.log2(np.abs(Rs[1:-1,I,I]))
    LEs = _.mean(axis=0) / nstep
    LEs = 2**LEs
    return LEs


def nilss(Cinv, d, R, b):
    # solve the nilss problem
    nseg, nus = d.shape
    D, E, Einv = np.nan * np.empty([3,nseg,nus,nus])
    RT = np.swapaxes(R,1,2)

    for i in range(nseg-1):
        D[i] = R[i+1] @ Cinv[i]

    for i in range(1, nseg):
        E[i] = D[i-1] @ RT[i] + Cinv[i]

    y, lbd, a = np.nan * np.empty([3, nseg, nus])
    for i in range(1, nseg):
        y[i] = D[i-1] @ d[i-1] - Cinv[i] @ d[i] - b[i]

    for i in range(2, nseg):
        tp = np.linalg.solve(E[i-1].T, D[i-1].T).T # use the new E! do not use Einv!!!
        E[i] -= tp @ D[i-1].T
        y[i] += tp @ y[i-1]

    lbd[nseg-1] = np.linalg.solve(E[nseg-1], y[nseg-1]) 
    for i in range(nseg-2, 0, -1):
        lbd[i] = np.linalg.solve(E[i], (D[i].T @ lbd[i+1] + y[i]))

    a[0] = D[0].T @ lbd[1] - Cinv[0] @ d[0]
    for i in range(1, nseg-1):
        a[i] = D[i].T @ lbd[i+1] - Cinv[i] @ (lbd[i] + d[i])
    a[nseg-1] = - Cinv[nseg-1] @ (lbd[nseg-1] + d[nseg-1])

    # compare
    kk = Cinv.shape[0]
    Cinv = block_diag(*Cinv)
    d = np.ravel(d) 
    B = np.eye((kk-1)*nus, kk*nus, k=nus)
    B[:, :-nus] -= block_diag(*R[1:])
    b = np.ravel(b[1:])
    
    lbdd = np.linalg.solve(-B @ Cinv @ B.T, B @ Cinv @ d + b)
    aa = -Cinv @ (B.T @ lbdd + d)
    aa = aa.reshape([kk, nus])
    temp1 = a - aa
    temp2 = np.ravel(lbd[1:]) - lbdd
    set_trace()
    return a


def tan2nd(rini, u, psi, w, vt):
    r = rini
    for i in range(nstep):
        fu, _ = fufs(u[i])
        fuu, fsu = fuufsu(u[i])
        rn = fu @ r + psi[i+1] * fsu @ w[i] + fuu @ vt[i] @ w[i]
        r = rn
    return r


def flr(nseg, W):
    # shadowing contribution and first order tangent
    Cinvs = np.nan * np.empty([nseg, nus, nus])
    dwvstars, dwvtstars, dwJus = np.nan * np.empty([3, nseg, nus])
    dvstarJus, dvtstarJus = np.nan * np.empty([2, nseg])
    Rs = np.nan * np.empty([nseg+1, nus, nus]) # R[0] is at t0, R[K] at T, but both not used
    Q = np.nan * np.empty([nseg+1, nc, nus]) 
    bs, bts = np.nan * np.empty([2, nseg+1, nus])
    vstars, vtstars = np.nan * np.empty([2, nseg, nstep+1, nc]) # only for debug 
    ws = np.nan * np.empty([nseg, nstep+1, nc, nus]) # only for debug and illustration

    u0 = preprocess()
    u, Ju, psi, Javg = primal(u0, nseg, W) # notice that u0 is changed

    Q[0], q, qt = Q0q0()
    Rs[0] = np.eye(nus)
    for k in range(nseg):
        Cinvs[k], dwvstars[k], dwvtstars[k], dwJus[k], dvstarJus[k], dvtstarJus[k],\
                ws[k], vstars[k], vtstars[k]\
                = inner_products(u[k], Ju[k], Q[k], q, psi[k], qt)
        Q[k+1], Rs[k+1], q, bs[k+1], qt, bts[k+1] = renormalize(ws[k,-1], vstars[k,-1], vtstars[k,-1])

    LEs = getLEs(Rs)
    aa = nilss(Cinvs, dwvstars, Rs[:-1], bs[:-1])
    aat = nilss(Cinvs, dwvtstars, Rs[:-1], bts[:-1])
    sc = ((dwJus * aa).sum() + dvstarJus.sum()) / (nseg * nstep) # shadowing contribution
    v = vstars + (ws*aa[:,newaxis,newaxis,:]).sum(-1) 
    vt = vtstars + (ws*aat[:,newaxis,newaxis,:]).sum(-1) 

    # unstable contribution and second order tangent
    rend = np.nan * np.empty([nseg, nc, nus])
    Rinv = np.linalg.inv(Rs)
    ucs = np.nan * np.empty([nseg])
    rini = np.zeros([nc, nus])
    for k in range (nseg):
        rend[k] = tan2nd(rini, u[k], psi[k], ws[k], vt[k]) # run second order tangent solver
        ucs[k] = (Rinv[k+1] @ Q[k+1].T @ rend[k]).trace() / nstep # the unstable contribution of k
        rini = (rend[k] - Q[k+1] @ Q[k+1].T @ rend[k]) @ Rinv[k+1] # renormalization
    uc = ucs.mean()

    return Javg, sc, uc, u, v, (Ju*v).sum(-1), LEs, vt
