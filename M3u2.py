# solonoid map, need to debug my linear response algorithm with this map
from __future__ import division
import numpy as np
from numpy import newaxis, sin, cos
import shutil
import sys
import itertools
from pdb import set_trace


nstep = 10 # step per segment
nus = 2 # u in paper, number of homogeneous tangent solutions
nc = 3 # M in papaer, dimension of phase space
nseg_ps = 100
nseg_dis = 20 # segments to discard, not even for Javg
R = 1.0 # the R in Patrick' paper
rho = 0.25 # the epsilong on Patrick's paper


def fJJu(x):
    r, t, z = x
    rn = R + 0.05*(r - R) + 0.1 *cos(t)
    tn = (t * 2 * r + rho*sin(t)) % (2*np.pi)
    zn = 0.05 * z + 0.1 * sin(t)
    J = r
    Ju = np.array([1, 0, 0])
    return np.array([rn, tn, zn]), J, Ju


def fufs(x):
    r, t, z = x
    fu = np.array([ [0.05, -0.1*sin(t), 0],
                    [0, 2+rho*cos(t), 0],
                    [0, 0.1*cos(t), 0.05]])
    fs = np.array([0, sin(t), 0]) # parameter is rho
    return fu, fs
