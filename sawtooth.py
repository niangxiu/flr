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


nstep = 1000000
thetas = np.empty(nstep)
thetas[0] = 1.2
starttime = time.time()
A = 0.5
B = 2

for i in range(nstep - 1):
    # thetas[i+1] = (thetas[i] * 2 + A*np.sin(thetas[i]) + B*(np.random.rand()-0.5)) % (2*np.pi)
    thetas[i+1] = (thetas[i] * 2 + A*np.sin(thetas[i])) % (2*np.pi)
plt.hist(thetas, 25)
plt.savefig('thetas distribution')

endtime = time.time()
print('time elapsed in seconds:', endtime-starttime)
