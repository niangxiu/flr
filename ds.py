# the file for expressions of the dynamical system
# sawtooth map
from __future__ import division
import numpy as np
from numpy import newaxis, sin, cos
import shutil
import sys
import itertools
from pdb import set_trace


nstep = 10 # step per segment
nus = 1 # u in paper, number of homogeneous tangent solutions
nc = 1 # M in papaer, dimension of phase space
nseg_ps = 100
nseg_dis = 20 # segments to discard, not even for Javg
prm = 0 # the epsilong on Patrick's paper
W  = 5


def fJJu(x):
    xn = (2 * x + prm * sin(x)) % (2*np.pi)
    J = cos(x)
    Ju = -sin(x)
    return np.array([xn]), J, np.array([Ju])


def fufs(x):
    fu = 2 + prm * cos(x)
    fs = sin(x)
    return np.array([[fu]]), np.array([fs])


def fuufsu(x):
    fuu = -prm * sin(x)
    fsu = -cos(x)
    return np.array([[[fuu]]]), np.array([[fsu]])
