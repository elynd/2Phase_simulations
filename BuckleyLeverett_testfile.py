# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 14:35:36 2016

@author: eric
"""
import numpy as np
from scipy.linalg import solve as solve
from pylab import *
import scipy.interpolate
import math
import matplotlib.pyplot as plt  
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix
from scipy.optimize import newton_krylov
from scipy.optimize import broyden1
import matplotlib.animation as animation
import time
from PIL import Image
import os
from Buckley_Leverett import *
#Buckley Leverett test file
q=1.
A=1
time=0.001

permeability=.01
###Fluid Properties
fluid_properties=np.array([1.,10.])
L=1
#self.mu_w=fluid_properties[0]; self.mu_o=fluid_properties[1]; self.B_w=fluid_properties[2]; self.B_o=fluid_properties[3]
###Reservoir properties
reservoir_properties=np.array([.1,.2,.7])
#rel_perm_params=np.array([k_wr,k_nwr,lam_1,lam_2])
rel_perm_params=np.array([.35,.95,3.,2.])
BL1=Buckley_Leverett(fluid_properties,reservoir_properties,rel_perm_params,q,A,permeability,L,time)
print(BL1.x_wf,BL1.S_wf,BL1.t_bt)
BL1.plotter()