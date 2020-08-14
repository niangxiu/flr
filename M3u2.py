# I am trying to find a discrete chaotic system with M=3 and u=2 as an example.
from __future__ import division
import numpy as np
from numpy import newaxis
import shutil
import sys
import itertools
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pdb import set_trace
import time


def duffing(nstep):
    a = 2.75
    b = 0.15
    x = np.zeros(nstep)
    y = np.zeros(nstep)
    x[0] = 1
    for n in range(nstep - 1):
        x[n+1] = y[n]
        y[n+1] = -b * x[n] + a* y[n] - y[n]**3
    # return np.dstack((x, y))
    return x, y


def Bogdanov(nstep):
    eps = 0
    k = 1.2
    mu = 0
    x = np.zeros(nstep)
    y = np.zeros(nstep)
    x[0] = 0.232
    for n in range(nstep - 1):
        y[n+1] = y[n] + eps*y[n] + k*x[n]*(x[n]-1) + mu*x[n]*y[n]
        x[n+1] = x[n] + y[n+1] 
    return x, y

def LorenzEuler(nstep):
    beta = 8/3
    rho = 28.0
    sigma = 10.0
    dt = 0.02
    x, y, z = np.zeros((3, nstep))
    x[0] = 10
    # f = np.array([sigma*(y-x), x*(rho-z)-y, x*y-beta*z])
    for n in range(nstep-1):
        x[n+1] = x[n] + sigma*(y[n]-x[n]) * dt  
        y[n+1] = y[n] + (x[n]*(rho-z[n])-y[n]) * dt
        z[n+1] = z[n] + (x[n]*y[n]-beta*z[n]) * dt
    return x,z

nstep = 1000000
starttime = time.time()
# x, y = duffing(nstep)
# x, y = Bogdanov(nstep)
x, y = LorenzEuler(nstep)

plt.plot(x, y, '.', markersize=1)
plt.savefig('phase.png')

endtime = time.time()
print('time elapsed in seconds:', endtime-starttime)
