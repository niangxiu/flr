# the file for expressions of the dynamical system
# the modified solonoid map
from __future__ import division
import numpy as np
from numpy import newaxis, sin, cos
import shutil
import sys
import itertools
from pdb import set_trace


nstep = 20 # step per segment
nus = 2 # u in paper, number of homogeneous tangent solutions
nc = 3 # M in papaer, dimension of phase space
nseg_ps = 100
nseg_dis = 20 # segments to discard, not even for Javg
prm = 0.1 # the epsilon on Patrick's paper
A = 8

def fJJu(x):
    r, t, p = x
    rn = 0.05*r + 0.1*cos(A*t) - 0.1*sin(5*p)
    tn = (t * 2 + prm * (1+r) * sin(A*t)) % (2*np.pi)
    pn = (p * 3 + prm * (1+r) * cos(2*p)) % (2*np.pi)
    J = r
    Ju = np.array([1, 0, 0])
    return np.array([rn, tn, pn]), J, Ju


def fufs(x):
    r, t, p = x
    fu = np.array([ [0.05, -0.1*A*sin(A*t), -0.5*cos(5*p)],
                    [prm*sin(A*t), 2 + prm*(1+r)*A*cos(A*t), 0],
                    [prm*cos(2*p), 0, 3 - prm*(1+r)*2*sin(2*p)]])
    fs = np.array([0, (1+r)*sin(A*t), (1+r)*cos(2*p)]) 
    return fu, fs


def fuufsu(x):
    r, t, p = x
    fuu = np.array([[[0,0,0], [0,-0.1*A*A*cos(A*t),0], [0,0,2.5*sin(5*p)] ],
                    [[0,prm*A*cos(A*t),0], [prm*A*cos(A*t),-prm*(1+r)*A*A*sin(A*t),0], [0,0,0] ],
                    [[0,0,-2*prm*sin(2*p)], [0,0,0], [-prm*2*sin(2*p),0,-prm*(1+r)*4*cos(2*p)]]])
    fsu = np.array([[0, 0, 0],
                    [sin(A*t), (1+r)*A*cos(A*t), 0],
                    [cos(2*p), 0, -(1+r)*2*sin(2*p)]])
    return fuu, fsu
