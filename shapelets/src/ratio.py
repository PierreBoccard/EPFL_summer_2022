import sys
from termios import IEXTEN

import numpy as np
import copy
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') 

import os
import astropy.io.fits as pyfits  # open / write FITS files
from PIL import Image  # images manipulation
import math
import pickle

nmax = np.array([4,5,5,8,8,8,8,10,12,12,15])

g1 = np.array([2.557,2.777,2.952,2.545,2.889,2.953,2.610,2.531,2.87,2.839,2.704])

g2 = np.array([0.698,0.495,0.39,0.706,0.435,0.386,0.684,0.73,0.468,0.478,0.635])
 
g3 = np.array([0.497,0.425,0.364,0.492,0.378,0.363,0.435,0.473,0.366,0.388,0.387])

g4 = np.array([0.182,0.229,0.213,0.193,0.222,0.216,0.183,0.186,0.206,0.213,0.183])

g1g2 = g1/g2
g1g3 = g1/g3
g1g4 = g1/g4


plt.figure()
plt.scatter(nmax,g1g2,label=r'$G1/G2$',color = 'g')
plt.scatter(nmax,g1g3,label=r'$G1/G3$',color = 'r')
plt.scatter(nmax,g1g4,label=r'$G1/G4$',color = 'b')
plt.legend()
plt.grid()
plt.xlabel(r'$n_{max}$')
plt.savefig('ratio.jpg')
plt.close()

