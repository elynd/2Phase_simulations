
# -*- coding: utf-8 -*-
""" Created on Thu Aug 27 17:20:47 2015 
@author: Eric
"""
#from SFEM_2D_nonbanded import SFEM_2D
#from IGA_2D_2Phase import IGA_2D_2Phase
from IGA_1D_2Phase_implicit import IGA_1D_2Phase_implicit
from IGA_1D_2Phase_EXPLICIT import IGA_1D_2Phase_EXPLICIT
from Buckley_Leverett import Buckley_Leverett
#from CMG_read import CMG_read
import numpy as np
#import math as math
import IGA_refine
import IGA_pre_processing as Ippre
import matplotlib
import time
#from fractions import Fraction

#matplotlib.use('pgf')
#pgf_with_custom_preamble = {
#    "font.family": "serif",   # use serif/main font for text elements
#    "text.usetex": True,    # use inline math for ticks
#    "pgf.rcfonts": False,   # don't setup fonts from rc parameters
#}
#matplotlib.rcParams.update(pgf_with_custom_preamble)
import matplotlib.pyplot as plt
#from matplotlib.backends.backend_pgf import FigureCanvasPgf
#matplotlib.backend_bases.register_backend('pdf', FigureCanvasPgf)
#import matplotlib

"""
Sets sig figs in outputs
"""
#np.set_printoptions(precision=6)
"""
Data for IGA
"""
""" 
Multiplicites
"""
multiplicity_1=1
multiplicity_2=1
"""
Order
#"""
order_1=1
order_2=1
"""
Number elements
"""

number_elements_1=1
number_elements_2=20
"""
Patch connectvity data

Simply put 0 if single patch
"""    
patch_connect_info=0 

"""
Creating object with stored infor for knot vectors and basis function arrays
"""
mesh_data_1=Ippre.patch_data(order_1,number_elements_1,multiplicity_1,patch_connect_info,1)   
mesh_data_2=Ippre.patch_data(order_2,number_elements_2,multiplicity_2,patch_connect_info,1)   
 
                                
"""
Define coarsest mesh
"""
B=np.array([[0,0,1],[1,0,1]])

"""
Refine coarsest control net
"""         
control_net_1=B         
control_net_2=IGA_refine.Line_refine(mesh_data_1,mesh_data_2,B).B_new
 
      
"""
Reservoir and fluid Parameters
"""

###permeability tensors
permeability=np.array([.01])     

###Reservoir properties: Porosity, pore distribution index, S_wirr, S_m (water saturation at non-wetting irreducible saturation),k_nwr (non wetting relative permability at S_wirr),c_o,c_w,c_r
S_m=.7
S_wirr=.2
reservoir_properties=np.array([.1,1.,S_wirr,S_m]).astype('float')
#rel_perm_params=np.array([k_wr,k_nwr,lam_1,lam_2])
rel_perm_params=np.array([.35,.95,3,2])
###Fluid properties: Viscosity,B_w,B_o
fluid_properties=np.array([1.,10.]).astype('float')
BL1=Buckley_Leverett(fluid_properties,np.array([.1,.2,.7]),rel_perm_params,1.,1.,.01,1.,.02)
t_bt=BL1.t_bt
#print(t_bt)
"""
Samples times (seconds)
""" 
times=np.linspace(0,t_bt*1.1,number_elements_2*1.5)/1.

""" 
Initial conditions
"""
initial_pressure_1=np.ones(len(control_net_1))*1
initial_saturation_1=np.ones(len(control_net_1))*1.
initial_conditions_1=np.hstack((initial_pressure_1,initial_saturation_1))

initial_pressure_2=np.ones(len(control_net_2))*0
initial_saturation_2=np.ones(len(control_net_2))*S_wirr
initial_conditions_2=np.hstack((initial_pressure_2,initial_saturation_2))
"""
Boundary conditions

For multiple patch systems, will need to make additional file to compile arrays with info for boundary conditions
"""
nuemann_boundaries=None            
nuemann_values=None
dirichlet_boundaries=None
dirichlet_values=None

#Use 1D array in 1D case, 2D array in 2D case
source_locations_flux_2=np.array([mesh_data_2.patch_global_2D(0)[0]])
source_values_flux_2=np.array([1])*1.
source_locations_pressure_2=np.array([mesh_data_2.patch_global_2D(0)[-1]])
source_values_pressure_2=np.array([1.])*0
boundary_saturation_location_2=np.array([mesh_data_2.patch_global_2D(0)[0]]) #,mesh_data_2.patch_global_2D(0)[-1]])
boundary_saturation_values_2=np.array([S_m]) #,S_wirr])


"""
Running analysis
"""
IGA_a_mesh_2=IGA_1D_2Phase_implicit(mesh_data_2,control_net_2,permeability,reservoir_properties,rel_perm_params,fluid_properties,times, \
initial_pressure_2,initial_saturation_2,nuemann_boundaries,nuemann_values,dirichlet_boundaries,dirichlet_values,source_locations_flux_2, \
source_values_flux_2,source_locations_pressure_2,source_values_pressure_2,boundary_saturation_location_2,boundary_saturation_values_2,1)
##
#IGA_a_mesh_2=IGA_1D_2Phase_EXPLICIT(mesh_data_2,control_net_2,permeability,reservoir_properties,rel_perm_params,fluid_properties,times, \
#initial_pressure_2,initial_saturation_2,nuemann_boundaries,nuemann_values,dirichlet_boundaries,dirichlet_values,source_locations_flux_2, \
#source_values_flux_2,source_locations_pressure_2,source_values_pressure_2,boundary_saturation_location_2,boundary_saturation_values_2,1)

filename='Order'+str(order_2)+'_e'+str(number_elements_2)+'.mp4'
IGA_a_mesh_2.anim_saturation(filename)




