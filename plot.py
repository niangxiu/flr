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
from M3u2 import *
import M3u2
from lra import nilss, primal


def wrapped_nilss(nseg, n_repeat): 
    arguments = [(nseg,) for i in range(n_repeat)]
    if n_repeat == 1:
        results = [nilss(*arguments[0])]
    else:
        with Pool(processes=4) as pool:
            results = pool.starmap(nilss, arguments)
    Javg_, grad_, *_ = zip(*results)
    print('rho, Javg, grad')
    [print(M3u2.rho, Javg, grad) for Javg, grad in zip(Javg_, grad_)]
    return np.array(Javg_), np.array(grad_)


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
            Javgs[i], grads[i] = wrapped_nilss(nseg, n_repeat)
        pickle.dump((Javgs, grads, nsegs), open("change_T.p", "wb"))
    
    plt.semilogx(nsegs, grads, 'k.')
    plt.tight_layout()
    plt.savefig('T_grad_rho.png')
    plt.close()
    x = np.array([nsegs[0], nsegs[-1]])
    plt.loglog(nsegs, np.std(grads, axis=1), 'k.')
    plt.loglog(x, x**-0.5, 'k--')
    plt.xlabel('$T$')
    plt.ylabel('std $\left( \, J_{avg} \, \\right)$')
    plt.tight_layout()
    plt.savefig('T_grad_std.png')

        
def change_rho():
    # grad for different rho
    n_repeat = 1
    nseg = 10
    rhos = np.linspace(-1.0, -1.5, 6)
    Javgs   = np.empty(rhos.shape[0])
    grads   = np.empty(rhos.shape[0])
    for i, rho in enumerate(rhos):
        M3u2.rho = rho
        Javgs[i], grads[i] = wrapped_nilss(nseg, n_repeat)


def all_info():
    # generate all info
    nseg = 100
    Javg, grad, u, v, Juv, LEs = nilss(nseg)
    plt.plot(u[:,:,0].reshape(-1), u[:,:,1].reshape(-1), '.', markersize=1)
    plt.savefig('trajectory.png')
    plt.close()
    print('Javg, grad = ', Javg, grad)
    print('Lyapunov exponenets = ', LEs)
    _, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(13,12))
    vn = v[:,:,0].reshape(-1)
    ax1.plot(np.arange(vn.shape[0]),vn)
    ax1.set_title('v norm')
    Juv = Juv[:,:].reshape(-1)
    ax2.plot(np.arange(vn.shape[0]),Juv)
    ax2.set_title('Ju @ v')
    ax3.scatter(u[:,:,0], v[:,:,0])
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
    # change_rho()
    all_info()
    # trajectory()
    print('rho=', M3u2.rho)
    endtime = time.time()
    print('time elapsed in seconds:', endtime-starttime)
