# expressions of the dynamical system: modified solonoid map with M=11 and u=10
from __future__ import division
import numpy as np
from numpy import newaxis, sin, cos
import shutil
import sys
import itertools
from pdb import set_trace


nstep = 20 # step per segment
nus = 4 # u in paper, number of homogeneous tangent solutions
nc = 5 # M in papaer, dimension of phase space
nseg_ps = 100
nseg_dis = 100 # segments to discard, not even for Javg
prm = 0.1 # the epsilon on Patrick's paper
A = 5
ii = list(range(1,nc))


def fJJu(x):
    xn = np.zeros(nc)
    xn[0] = 0.05*x[0] + 0.1*cos(A*x[ii]).sum() + prm
    xn[ii] = (2*x[ii] + prm*(1+x[0]) * sin(2*x[ii])) % (2*np.pi)

    J = x[0]**3 + 0.005 * ((x[ii] - np.pi)**2).sum()
    Ju = np.zeros(x.shape)
    Ju[0] = 3* x[0]**2
    Ju[ii] = 0.005 * 2 * (x[ii] - np.pi)
    return xn, J, Ju


def fufs(x):
    fu = np.zeros([nc,nc])
    fu[0,0] = 0.05
    fu[0,ii] = -0.1*A*sin(A*x[ii])
    fu[ii,0] = prm * sin(2*x[ii])
    fu[ii,ii] = 2 + prm*(1+x[0])*2*cos(2*x[ii])

    fs = np.zeros(nc)
    fs[0] = 1
    fs[ii] = (1+x[0]) * sin(2*x[ii])
    return fu, fs


def fuufsu(x):
    fuu = np.zeros([nc,nc,nc])
    fuu[0,ii,ii] = -0.1 * A**2 * cos(A*x[ii])
    fuu[ii,0,ii] = 2 * prm * cos(2*x[ii])
    fuu[ii,ii,0] = 2 * prm * cos(2*x[ii])
    fuu[ii,ii,ii]= -prm * (1+x[0]) * 4 * sin(2*x[ii])

    fsu = np.zeros([nc,nc])
    fsu[ii,0] = sin(2*x[ii])
    fsu[ii,ii] = (1+x[0]) * 2* cos(2*x[ii])
    return fuu, fsu
