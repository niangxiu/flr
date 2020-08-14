from __future__ import division
import numpy as np
from numpy import newaxis
import shutil
import sys
import itertools
from pdb import set_trace


nstep = 20 # step per segment
nus = 2 # M in paper, number of homogeneous tangent solutions
nc = 2 # m in papaer, dimension of phase space
nseg_ps = 1100
nseg_dis = 100 # segments to discard, not even for Javg
A = 0
B = 0


def fJJu(theta):
    xx = B * (np.random.rand(2) - 0.5) # a random input
    theta_next = (theta * 2 + A*np.sin(theta) + xx) % (2*np.pi)
    J = np.cos(theta[0])
    Ju = np.array([-np.sin(theta[0]),0])
    return theta_next, J, Ju


def fufs(theta):
    fu = np.diag(2 + A*np.cos(theta)) 
    fs  = np.sin(theta) # parameter is A
    return fu, fs
