# this is the file for non-autonomous lorenz system with a periodic excitation.
from __future__ import division
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shutil
import sys
import os
import time
import pickle
import itertools
from multiprocessing import Pool, current_process
from pdb import set_trace
from ds import *
import ds
from lra import lra, primal

plt.rc('axes', labelsize='x-large',  labelpad=12)
plt.rc('xtick', labelsize='x-large')
plt.rc('ytick', labelsize='x-large')
plt.rc('legend', fontsize='x-large')
plt.rc('font', family='sans-serif')


def wrapped_lra(nseg, n_repeat): 
    arguments = [(nseg,) for i in range(n_repeat)]
    if n_repeat == 1:
        results = [lra(*arguments[0])]
    else:
        with Pool(processes=4) as pool:
            results = pool.starmap(lra, arguments)
    Javg_, sc_, uc_, *_ = zip(*results)
    print('prm, Javg, sc, uc, grad')
    [print('{:.3e}, {:.3e}, {:.3e}, {:.3e}, {:.3e}'.format(ds.prm, Javg, sc, uc, sc-uc)) \
            for Javg, sc, uc in zip(Javg_, sc_, uc_)]
    return np.array(Javg_), np.array(sc_), np.array(uc_)


def converge_T():
    # convergence of gradient to different trajectory length
    try:
        Javgs, grads, nsegs = pickle.load( open("change_T.p", "rb"))
    except FileNotFoundError:
        n_repeat = 8
        # K_segment_ = np.array([2, 5, 1e1, 2e1, 5e1, 1e2, 2e2, 5e2, 1e3], dtype=int) 
        nsegs = np.array([2, 5, 1e1, 2e1, 5e1, 1e2], dtype=int) 
        Javgs = np.empty((nsegs.shape[0], n_repeat))
        grads = np.empty((nsegs.shape[0], n_repeat))
        for i, nseg in enumerate(nsegs):
            print('\nK=',nseg)
            Javgs[i], grads[i] = wrapped_lra(nseg, n_repeat)
        pickle.dump((Javgs, grads, nsegs), open("change_T.p", "wb"))
    
    plt.semilogx(nsegs, grads, 'k.')
    plt.tight_layout()
    plt.savefig('T_grad_prm.png')
    plt.close()
    x = np.array([nsegs[0], nsegs[-1]])
    plt.loglog(nsegs, np.std(grads, axis=1), 'k.')
    plt.loglog(x, x**-0.5, 'k--')
    plt.xlabel('$T$')
    plt.ylabel('std $\left( \, J_{avg} \, \\right)$')
    plt.tight_layout()
    plt.savefig('T_grad_std.png')

        
def change_prm():
    # grad for different prm
    n_repeat = 1
    nseg = 1000
    prms = np.linspace(0, 1, 41)
    A = 0.05
    Javgs, sc, uc   = np.empty([3,prms.shape[0]])
    try:
        prms, Javgs, grads  = pickle.load( open("data.p", "rb"))
    except FileNotFoundError:
        for i, prm in enumerate(prms):
            ds.prm = prm
            Javgs[i], sc[i], uc[i] = wrapped_lra(nseg, n_repeat)
        grads = sc - uc
        pickle.dump((prms, Javgs, grads), open("data.p", "wb"))
    plt.plot(prms, Javgs, 'k.', markersize=6)
    for prm, Javg, grad in zip(prms, Javgs, grads):
        plt.plot([prm-A, prm+A], [Javg-grad*A, Javg+grad*A], color='grey', linestyle='-')
    plt.ylabel('$\\rho(\Phi)$')
    plt.xlabel('$\gamma$')
    plt.tight_layout()
    plt.savefig('prmobj.png')


def all_info():
    # generate all info
    nseg = 1000
    Javg, sc, uc, u, v, Juv, LEs, vt = lra(nseg)
    # _, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8,8))
    # ax1.plot(u[:,:,1].reshape(-1), u[:,:,0].reshape(-1), '.', markersize=1)
    # plt.xlabel('x0')
    # plt.ylabel('x1')
    # ax2.plot(u[:,:,2].reshape(-1), u[:,:,0].reshape(-1), '.', markersize=1)
    # plt.xlabel('x2')
    # plt.ylabel('x1')
    # ax3.plot(u[:,:,1].reshape(-1), u[:,:,2].reshape(-1), '.', markersize=1)
    # plt.xlabel('x2')
    # plt.ylabel('x1')
    plt.hist(u[:,:,0].reshape(-1))
    plt.tight_layout()
    plt.savefig('trajectory.png')
    plt.close()
    print('Javg, sc, uc, grad = ', '{:.3e}, {:.3e}, {:.3e}, {:.3e}'.format(Javg, sc, uc, sc-uc))
    print('Lyapunov exponenets = ', LEs)
    _, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(13,12))
    vn = v[:,:,0].reshape(-1)
    ax1.plot(np.arange(vn.shape[0]),vn)
    ax1.set_title('v norm')
    Juv = Juv[:,:].reshape(-1)
    ax2.plot(np.arange(vn.shape[0]),Juv)
    ax2.set_title('Ju @ v')
    ax3.scatter(u[:,:,0], v[:,:,0])
    vtn = vt[:,:,0].reshape(-1)
    ax4.plot(np.arange(vtn.shape[0]),vtn)
    ax4.set_title('vtilde norm')
    plt.tight_layout()
    plt.savefig('v norm.png')
    plt.close()


def trajectory():
    np.random.seed()
    u0 = (np.random.rand() - 0.5) * 100
    u, f, J, Ju = primal(u0, t0=0, nstep = 10000)
    plt.plot(u[:,0].T, u[:,2].T,'k')
    plt.savefig('trajectory.png')
    plt.close()
    

if __name__ == '__main__': # pragma: no cover
    starttime = time.time()
    # converge_T()
    change_prm()
    # all_info()
    # trajectory()
    print('prm=', ds.prm)
    endtime = time.time()
    print('time elapsed in seconds:', endtime-starttime)
