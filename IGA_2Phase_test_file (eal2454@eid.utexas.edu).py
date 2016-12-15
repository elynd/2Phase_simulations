
# -*- coding: utf-8 -*-
""" Created on Thu Aug 27 17:20:47 2015 
@author: Eric
"""
#from SFEM_2D_nonbanded import SFEM_2D
#from IGA_2D_2Phase import IGA_2D_2Phase
from IGA_1D_2Phase_implicit import IGA_1D_2Phase_implicit
from IGA_1D_2Phase_EXPLICIT import IGA_1D_2Phase_EXPLICIT

#from CMG_read import CMG_read
import numpy as np
#import math as math
import IGA_refine
import IGA_pre_processing as Ippre
import matplotlib
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
S_m=.9
S_wirr=.1
reservoir_properties=np.array([.1,1.,S_wirr,S_m]).astype('float')
#rel_perm_params=np.array([k_wr,k_nwr,lam_1,lam_2])
rel_perm_params=np.array([1,1,2,2])
###Fluid properties: Viscosity,B_w,B_o
fluid_properties=np.array([1.,1.,1.,1.]).astype('float')

"""
Samples times (seconds)
""" 
times=np.linspace(0,.07,120)/1.

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
boundary_saturation_location_2=np.array([mesh_data_2.patch_global_2D(0)[0]])
#,mesh_data_2.patch_global_2D(0)[-1]])
boundary_saturation_values_2=np.array([S_m])
#,S_wirr])


"""
Running analysis
"""
IGA_a_mesh_2=IGA_1D_2Phase_implicit(mesh_data_2,control_net_2,permeability,reservoir_properties,rel_perm_params,fluid_properties,times, \
initial_pressure_2,initial_saturation_2,nuemann_boundaries,nuemann_values,dirichlet_boundaries,dirichlet_values,source_locations_flux_2, \
source_values_flux_2,source_locations_pressure_2,source_values_pressure_2,boundary_saturation_location_2,boundary_saturation_values_2,1)
#
#IGA_a_mesh_2=IGA_1D_2Phase_EXPLICIT(mesh_data_2,control_net_2,permeability,reservoir_properties,rel_perm_params,fluid_properties,times, \
#initial_pressure_2,initial_saturation_2,nuemann_boundaries,nuemann_values,dirichlet_boundaries,dirichlet_values,source_locations_flux_2, \
#source_values_flux_2,source_locations_pressure_2,source_values_pressure_2,boundary_saturation_location_2,boundary_saturation_values_2,1)


IGA_a_mesh_2.anim_saturation('Order1_C0_20__xx.mp4')



"""
Saving plots
"""

a=IGA_a_mesh_2.plotter('saturations')
#a.save('pressures_1.gif', writer='imagemagick')
#plt.savefig('./figs/R_200_IGA_1.pgf')
#plt.savefig('./figs/R_200_IGA_1.png')
#plt.show()
#
#plt = IGA_a_mesh_1.plotter(5,1,1)
#plt.savefig('./figs/R_200_IGA_1.pgf')
#plt.savefig('./figs/R_200_IGA_1.png')
#plt.show()



"""
Creates vector of x,y points to be sampled in SFEM code
"""
n=20
test_xi1=np.linspace(.01,.99,n); test_eta1=np.linspace(.01,.99,n)

XI1,ETA1=np.meshgrid(test_xi1,test_eta1); test_xi1=XI1.ravel(); test_eta1=ETA1.ravel()

test_xi=np.empty(0); test_eta=np.empty(0); test_x=np.empty(0); test_y=np.empty(0); test_pnum=np.empty(0)

R=IGA_a_mesh_1.NURBS_bases(test_xi1,test_eta1,0)
B_patch_ind=IGA_a_mesh_1.patch_data.patch_global_2D(0).flatten()
x_patch=IGA_a_mesh_1.B[B_patch_ind,0]
y_patch=IGA_a_mesh_1.B[B_patch_ind,1]
x_pt=np.sum(R*x_patch,axis=1)
y_pt=np.sum(R*y_patch,axis=1)
test_x=np.append(test_x,x_pt)
test_y=np.append(test_y,y_pt)
test_xi=np.append(test_xi,test_xi1)
test_eta=np.append(test_eta,test_eta1)
test_pnum=np.append(test_pnum,0)
  


def L2_error_FD(object2,object1,test_x,test_y,test_xi,test_eta,n):
    """
    Finds L2 error between 2 meshes using node locations of finer mesh as test points. Object 1 gives the IGA solution, object 2 the FD solution
    """
    x=test_x
    y=test_y
    xi=test_xi
    eta=test_eta
#    x=object2.x
#    y=object2.y
    
#    for pnum in range(8):
    for pt_num in range(len(test_x)):
        pnum=np.floor(pt_num/n**2).astype('int')
        if pt_num==0 and pnum==0:
            ob_1_approx=(object1.approx_function(xi[pt_num],eta[pt_num],pnum))
            ob_2_approx=(object2.p_function(x[pt_num],y[pt_num]))
#                if ob_2_approx==-1:
#                    error=np.empty(0)
#                    continue
#                else:
            error=np.array([(ob_1_approx-ob_2_approx)**2])
             
        else:
            ob_1_approx=(object1.approx_function(xi[pt_num],eta[pt_num],pnum))
            ob_2_approx=(object2.p_function(x[pt_num],y[pt_num]))
#                if ob_2_approx==-1:
#                    continue
#                else:
            err=np.array([(ob_1_approx-ob_2_approx)**2])
            error=np.append(error,err)
            
    return np.sqrt(np.sum(error))
    


       
def L2_error_IGA(object_2,object_1,test_xi,test_eta,n):
    """
    Finds L2 error between 2 meshes using node locations of finer mesh as test points. Object 2 assumed to give the better solution
    """
    xi=test_xi
    eta=test_eta
#    x=object2.x
#    y=object2.y
    
#    for pnum in range(8):
    for pt_num in range(len(test_x)):
        pnum=np.floor(pt_num/n**2).astype('int')
        if pt_num==0 and pnum==0:
            ob_1_approx=(object_1.approx_function(xi[pt_num],eta[pt_num],pnum))
            ob_2_approx=(object_2.approx_function(xi[pt_num],eta[pt_num],pnum))
            error=np.array([(ob_2_approx-ob_1_approx)**2])
             
        else:
            ob_1_approx=(object_1.approx_function(xi[pt_num],eta[pt_num],pnum))
            ob_2_approx=(object_2.approx_function(xi[pt_num],eta[pt_num],pnum))
            err=np.array([(ob_2_approx-ob_1_approx)**2])
            error=np.append(error,err)
            
    return np.sqrt(np.sum(error))   
    
"""
Mean percent error using SFEM as ref
"""

#L2_1_FD=MPE_FD_v2(CMG_1,demo_64,test_x,test_y,test_xi,test_eta)
#L2_2_FD=MPE_FD_v2(CMG_2,demo_64,test_x,test_y,test_xi,test_eta)

#L2_1_SFEM=MPE_SFEM_v2(demo_64,demo_2,test_x,test_y)
#L2_2_SFEM=MPE_SFEM_v2(demo_64,demo_4,test_x,test_y)

#L2_1_IGA=MPE_IGA_v2(demo_64,IGA_a_mesh_1,test_x,test_y,test_xi,test_eta)
#L2_2_IGA=MPE_IGA_v2(demo_64,IGA_a_mesh_2,test_x,test_y,test_xi,test_eta)

"""
Mean percent error using IGA as ref
"""
#L2_1_FD=MPE_FD(CMG_1,IGA_a_mesh_5,test_x,test_y,test_xi,test_eta)
#L2_2_FD=MPE_FD(CMG_2,IGA_a_mesh_5,test_x,test_y,test_xi,test_eta)

#L2_1_IGA=MPE_IGA(IGA_a_mesh_5,IGA_a_mesh_1,test_xi,test_eta)
#L2_2_IGA=MPE_IGA(IGA_a_mesh_5,IGA_a_mesh_2,test_xi,test_eta)

"""
L2 error using IGA as ref
"""
L2_1_FD=L2_error_FD(CMG_1,IGA_a_mesh_5,test_x,test_y,test_xi,test_eta,n)
L2_2_FD=L2_error_FD(CMG_2,IGA_a_mesh_5,test_x,test_y,test_xi,test_eta,n)

L2_1_IGA=L2_error_IGA(IGA_a_mesh_5,IGA_a_mesh_1,test_xi,test_eta,n)
L2_2_IGA=L2_error_IGA(IGA_a_mesh_5,IGA_a_mesh_2,test_xi,test_eta,n)

dof_FD_1=len(CMG_1.x)
dof_FD_2=len(CMG_2.x)

dof_SFEM_1=len(demo_2.x)
dof_SFEM_2=len(demo_4.x)





dof_h=np.array([dof_IGA_1,dof_IGA_2])
dof_p=np.array([dof_IGA_1,dof_IGA_6,dof_IGA_7,dof_IGA_8])
dof_sfem=np.array([dof_SFEM_1,dof_SFEM_2,dof_SFEM_3,dof_SFEM_4])
dof_fd=np.array([dof_FD_1,dof_FD_2,dof_FD_3,dof_FD_4])
dof_mixed=np.array([dof_IGA_1,dof_IGA_14,dof_IGA_15,dof_IGA_16])


errs_h=np.array([L2_1_IGA,L2_2_IGA,L2_3_IGA,L2_4_IGA])
errs_p=np.array([L2_1_IGA,L2_6_IGA,L2_7_IGA,L2_8_IGA])
errs_sfem=np.array([L2_1_SFEM,L2_2_SFEM,L2_3_SFEM,L2_4_SFEM])
errs_fd=np.array([L2_1_FD,L2_2_FD,L2_3_FD,L2_4_FD])
errs_mixed=np.array([L2_1_IGA,L2_14_IGA,L2_15_IGA,L2_16_IGA])


coefsh=np.polyfit(np.log10(dof_h),np.log10(errs_h),1)
yh=10**(coefsh[0]*np.log10(dof_h)+coefsh[1])
coefsp=np.polyfit(np.log10(dof_p),np.log10(errs_p),1)
yp=10**(coefsp[0]*np.log10(dof_p)+coefsp[1])
coefsSFEM=np.polyfit(np.log10(dof_sfem),np.log10(errs_sfem),1)
ySFEM=10**(coefsSFEM[0]*np.log10(dof_sfem)+coefsSFEM[1])
coefsFD=np.polyfit(np.log10(dof_fd),np.log10(errs_fd),1)
yFD=10**(coefsFD[0]*np.log10(dof_fd)+coefsFD[1])
coefsMixed=np.polyfit(np.log10(dof_mixed),np.log10(errs_mixed),1)
yMixed=10**(coefsMixed[0]*np.log10(dof_mixed)+coefsMixed[1])
#m_h=Fraction(coefsh[0]).limit_denominator(1000)
#m_p=Fraction(coefsp[0]).limit_denominator(1000)
#m_SFEM=Fraction(coefsSFEM[0]).limit_denominator(1000)
#m_FD=Fraction(coefsFD[0]).limit_denominator(1000)


m_h=coefsh[0]
m_p=coefsp[0]
m_SFEM=coefsSFEM[0]
m_FD=coefsFD[0]
m_Mixed=coefsMixed[0]
"""
PLotting
"""
#l=Line2D([0,100],[0,100],linewidth=2,color='blue')
fig,ax=plt.subplots(1)
plt.loglog([dof_FD_1,dof_FD_2,dof_FD_3,dof_FD_4],[L2_1_FD,L2_2_FD,L2_3_FD,L2_4_FD],'m-d',ms=8)
#plt.loglog([dof_SFEM_1,dof_SFEM_2,dof_SFEM_3,dof_SFEM_4,dof_SFEM_5,dof_SFEM_6],[L2_1_SFEM,L2_2_SFEM,L2_3_SFEM,L2_4_SFEM,L2_5_SFEM,L2_6_SFEM],'b-o',ms=8)
plt.loglog([dof_SFEM_1,dof_SFEM_2,dof_SFEM_3,dof_SFEM_4],[L2_1_SFEM,L2_2_SFEM,L2_3_SFEM,L2_4_SFEM],'b-o',ms=8)

plt.loglog([dof_IGA_1,dof_IGA_2,dof_IGA_3,dof_IGA_4],[L2_1_IGA,L2_2_IGA,L2_3_IGA,L2_4_IGA],'r-x',ms=8,mew=2 )
plt.loglog([dof_IGA_1,dof_IGA_6,dof_IGA_7,dof_IGA_8],[L2_1_IGA,L2_6_IGA,L2_7_IGA,L2_8_IGA],'g-s',ms=8)
#plt.loglog([dof_IGA_1,dof_IGA_10,dof_IGA_11,dof_IGA_12,dof_IGA_13],[L2_1_IGA,L2_10_IGA,L2_11_IGA,L2_12_IGA,L2_13_IGA],'c-+',ms=8,mew=2)
plt.loglog([dof_IGA_1,dof_IGA_14,dof_IGA_15,dof_IGA_16],[L2_1_IGA,L2_14_IGA,L2_15_IGA,L2_16_IGA],'c-^',ms=8,mew=2)

plt.xlabel('Degrees of Freedom',fontsize=18)
plt.ylabel('L2 Error',fontsize=18)
axes = plt.gca()
#ax.add_line(l)
#plt.xlim([10, 10000])
plt.axis([10,1000,300,2000])
plt.gcf().subplots_adjust(bottom=0.15)  
plt.tick_params(labelsize=18)
#axes.set_aspect('equal')

#plt.title('L2 error vs. dof')
#plt.legend(['FD','SFEM','IGA: h refined','IGA: p refined','IGA: k refined','IGA:hpk refined'],loc='center',ncol=3,bbox_to_anchor=(.5, 1),fontsize=14)
plt.legend(['FD','SFEM','IGA: h','IGA: p','IGA: hpk'],loc=3,fontsize=14)

#axes.set_ylim([10,1000])
axes.grid('on','both')
#plt.axis([10,10000,100,1000])
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 20}
matplotlib.rc('font', **font)
plt.savefig('./figs/R_200_convergence_global.pgf')
plt.savefig('./figs/R_200_convergence_global.png')

