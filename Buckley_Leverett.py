# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 15:38:43 2016

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
from scipy.optimize import minimize

import matplotlib.animation as animation
import time
from PIL import Image
import os

class Buckley_Leverett():
    def __init__(self,fluid_properties,reservoir_properties,rel_perm_params,q,A,permeability,L,t):
        ###Fluid Properties
        self.mu_w=fluid_properties[0]; self.mu_o=fluid_properties[1]
        ###Reservoir properties
        self.k=permeability; self.phi=reservoir_properties[0]; self.S_wirr=reservoir_properties[1]; self.S_m=reservoir_properties[2]
        self.L=L #res length
        self.time=t
        ###Rel perm parameters
        self.k_wr=rel_perm_params[0]; self.k_nwr=rel_perm_params[1];self.lam_1=rel_perm_params[2];self.lam_2=rel_perm_params[3]
        
        self.q=q
        self.A=A
        
        self.S_wf,self.x_wf,self.t_bt=self.shock_front_sat_and_pos()

    def relative_permeability(self,S_w,deriv=0):
        """
        Finds relative permeabilities based off of the Brooks Corey model
        
        Sw= water saturation
        Sw_irr= irreducible water saturation
        lam=pore size distribution index 
        k_nwr= nonwetting phase relative permeability at irreducible wetting phase saturation
        S_m= wetting phase saturation corresponding to the critical nonwetting phase saturation      
        S_wi   Peters book has S_wi, but it may be a typo for S_wirr
        """
        Sw_irr=self.S_wirr; S_m=self.S_m
        Sw_norm=(S_w-Sw_irr)/(S_m-Sw_irr)
        
        krw=self.k_wr*(Sw_norm)**self.lam_1
        krw1=(self.k_wr*self.lam_1)*Sw_norm**(self.lam_1-1)*1/(S_m-Sw_irr)
        kro=self.k_nwr*(1-Sw_norm)**self.lam_2
        kro1=(self.k_nwr*self.lam_2)*(1-Sw_norm)**(self.lam_2-1)*(-1)/(S_m-Sw_irr)

        if deriv==0:
            return krw, kro
        else:
            return krw1,kro1
     
    def calc_water_fraction(self,Sw,deriv=0):
        k_rws,k_ros=self.relative_permeability(Sw)
        k_rws1,k_ros1=self.relative_permeability(Sw,1)
        fw=1./(1.+k_ros*self.mu_w/(self.mu_o*k_rws)) #fw
        if deriv==0:
            return fw
        else:
            return -(fw)**2*(self.mu_w/self.mu_o*(k_ros1*k_rws**-1+ -k_rws**-2*k_rws1*k_ros)) #dfw/dSw
        
    def shock_front_sat_and_pos(self,init_guess=.5):
        """
        """
        #Residual function for finding S_wf from tangent line on fw vs. Sw diagram
        S_wf=lambda Sw: np.abs(self.calc_water_fraction(Sw)/(Sw-self.S_wirr)-self.calc_water_fraction(Sw,1))
        S_wf=minimize(S_wf,init_guess,method='nelder-mead').x

        #Evaluating fw derivative at S_wf
        dfw_dSw_wf=self.calc_water_fraction(S_wf,1)
        #Current position of water front
        x_wf=self.q*self.time/self.phi/self.A*dfw_dSw_wf
        #Breakthrough time
        t_bt=self.L*(self.q/self.phi/self.A*dfw_dSw_wf)**-1
        return S_wf,x_wf,t_bt
    
    def sat_curve(self,pts=200):
         if self.x_wf>=self.L:
            Sw_behind_front=np.linspace(self.S_wf,self.S_m,pts)
            dfw_dSw=self.calc_water_fraction(Sw_behind_front,deriv=1)
            x_behind_front=self.q*self.time/self.phi/self.A*dfw_dSw
            ind=np.argmin(np.abs(x_behind_front-self.L))
            Sw_behind_front=Sw_behind_front[ind:]
            x_behind_front=x_behind_front[ind:]
            return x_behind_front,Sw_behind_front
         else:
#            print(self.x_wf)
#            print(np.linspace(self.S_wf,self.S_m,self.x_wf/self.L*pts))
            Sw_behind_front=np.linspace(self.S_wf,self.S_m,np.ceil(self.x_wf/self.L*pts))
#            print((self.x_wf/self.L*pts))
            dfw_dSw=self.calc_water_fraction(Sw_behind_front,deriv=1)
            x_behind_front=self.q*self.time/self.phi/self.A*dfw_dSw
#            print(Sw_behind_front)
            ind=np.argmin(np.abs(x_behind_front-self.L))
            Sw_behind_front=Sw_behind_front[ind:]
            x_behind_front=x_behind_front[ind:]

            x_ahead_front=np.linspace(self.x_wf,self.L,np.round((self.L-self.x_wf)/self.L*pts))
            Sw_ahead_front=np.ones(len(x_ahead_front))*self.S_wirr
            
            Sw=np.hstack((Sw_ahead_front,Sw_behind_front))
            Sw[-1]=self.S_m
            x=np.hstack((x_ahead_front[::-1],x_behind_front))
            x[-1]=0
            return x,Sw
            
    def plotter(self):
        x,Sw=self.sat_curve()
        plt.plot(x,Sw)
        plt.xlabel('Distance')
        plt.ylabel('Water Saturation')
        plt.grid('on')
        plt.axis([0,self.L,0,1])
        plt.show
  
