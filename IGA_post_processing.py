# -*- coding: utf-8 -*-
""" Created on Thu Aug 27 17:20:47 2015 
@author: Eric
"""
from SFEM_2D_nonbanded import SFEM_2D
from IGA_2D import IGA_2D
from CMG_read import CMG_read
import numpy as np
import math as math
import IGA_refine
import IGA_pre_processing as Ippre
import matplotlib
#from matplotlib.lines import Line2D                        

#from mpltools import annotation
from fractions import Fraction

matplotlib.use('pgf')
pgf_with_custom_preamble = {
    "font.family": "serif",   # use serif/main font for text elements
    "text.usetex": True,    # use inline math for ticks
    "pgf.rcfonts": False,   # don't setup fonts from rc parameters
}
matplotlib.rcParams.update(pgf_with_custom_preamble)
import matplotlib.pyplot as plt
#from matplotlib.backends.backend_pgf import FigureCanvasPgf
#matplotlib.backend_bases.register_backend('pdf', FigureCanvasPgf)
#import matplotlib
"""
Data for FD
"""

#R=105
#CMG_file_path_1='CMG_data (eal2454@eid.utexas.edu)/r_105_gs_10 Pressure Time 1905-01-01.txt'
#CMG_1=CMG_read(CMG_file_path_1)
#R_105_FD_1=CMG_1.plotter()
#plt.savefig('./figs/R_105_FD_1.pgf')
#plt.savefig('./figs/R_105_FD_1.png')
#
#
#CMG_file_path_2='CMG_data (eal2454@eid.utexas.edu)/r_105_gs_25 Pressure Time 1905-01-01.txt'
#CMG_2=CMG_read(CMG_file_path_2)
#plot2=CMG_2.plotter()
#plt.show()
#
#CMG_file_path_3='CMG_data (eal2454@eid.utexas.edu)/r_105_gs_50 Pressure Time 1905-01-01.txt'
#CMG_3=CMG_read(CMG_file_path_3)
#R_105_FD_3=CMG_3.plotter()
#plt.savefig('./figs/R_105_FD_3.pgf')
#plt.savefig('./figs/R_105_FD_3.png')
#
#CMG_file_path_4='CMG_data (eal2454@eid.utexas.edu)/r_105_gs_100 Pressure Time 1905-01-01.txt'
#CMG_4=CMG_read(CMG_file_path_4)
#plot4=CMG_4.plotter()
#plt.show()



#R=125
#CMG_file_path_1='CMG_data (eal2454@eid.utexas.edu)/r_125_gs_10 Pressure Time 1905-01-01.txt'
#CMG_1=CMG_read(CMG_file_path_1)
#R_125_FD_1=CMG_1.plotter()
#plt.savefig('./figs/R_125_FD_1.pgf')
#plt.savefig('./figs/R_125_FD_1.png')
#
#CMG_file_path_2='CMG_data (eal2454@eid.utexas.edu)/r_125_gs_25 Pressure Time 1905-01-01.txt'
#CMG_2=CMG_read(CMG_file_path_2)
##plot2=CMG_2.plotter()
##plt.show()
#
#CMG_file_path_3='CMG_data (eal2454@eid.utexas.edu)/r_125_gs_50 Pressure Time 1905-01-01.txt'
#CMG_3=CMG_read(CMG_file_path_3)
##plot3=CMG_3.plotter()
##plt.show()
#
#CMG_file_path_4='CMG_data (eal2454@eid.utexas.edu)/r_125_gs_100 Pressure Time 1905-01-01.txt'
#CMG_4=CMG_read(CMG_file_path_4)
##plot4=CMG_4.plotter()
##plt.show()

R=200
CMG_file_path_1='CMG_data (eal2454@eid.utexas.edu)/r_200_gs_10 Pressure Time 1905-01-01.txt'
CMG_1=CMG_read(CMG_file_path_1)
R_200_FD_1=CMG_1.plotter()
plt.savefig('./figs/R_200_FD_1.pgf')
plt.savefig('./figs/R_200_FD_1.png')

CMG_file_path_2='CMG_data (eal2454@eid.utexas.edu)/r_200_gs_25 Pressure Time 1905-01-01.txt'
CMG_2=CMG_read(CMG_file_path_2)
#plot2=CMG_2.plotter()
#plt.show()

CMG_file_path_3='CMG_data (eal2454@eid.utexas.edu)/r_200_gs_50 Pressure Time 1905-01-01.txt'
CMG_3=CMG_read(CMG_file_path_3)
#plot3=CMG_3.plotter()
#plt.show()

CMG_file_path_4='CMG_data (eal2454@eid.utexas.edu)/r_200_gs_100 Pressure Time 1905-01-01.txt'
CMG_4=CMG_read(CMG_file_path_4)
#plot4=CMG_4.plotter()



"""
Data for SFEM
"""

degree=1
  
def k_term(x,y):
    
    return np.ones(len(x))*10.**-10
    
def f_term(x,y):
    
    return np.ones(len(x))*0
    
bc_top=lambda x,y: 0
bc_bottom=lambda x,y: 0
bc_left=lambda x,y: 0
bc_right=lambda x,y: 0

bc_array=lambda x,y: np.array([bc_top(x,y),bc_bottom(x,y),bc_left(x,y),bc_right(x,y)])

source=-50E-7
#source=1.

corners=np.array([1,1,1,1])        

#R=105
#demo_2=SFEM_2D(degree,'Data-selected/s_curve_mi_2_R_1.05.g',k_term,f_term,corners,source) 
#plt = demo_2.plotter()
#plt.savefig('./figs/R_105_SFEM_1.pgf')
#plt.savefig('./figs/R_105_SFEM_1.png')
#
#demo_4=SFEM_2D(degree,'Data-selected/s_curve_mi_4_R_1.05.g',k_term,f_term,corners,source) 
##demo_4.plotter()
#
#demo_8=SFEM_2D(degree,'Data-selected/s_curve_mi_8_R_1.05.g',k_term,f_term,corners,source) 
#plt=demo_8.plotter()
##plt.show()
#plt.savefig('./figs/R_105_SFEM_3.pgf')
#plt.savefig('./figs/R_105_SFEM_3.png')
#
#demo_16=SFEM_2D(degree,'Data-selected/s_curve_mi_16_R_1.05.g',k_term,f_term,corners,source) 
##demo_16.plotter()
#
#demo_32=SFEM_2D(degree,'Data-selected/s_curve_mi_32_R_1.05.g',k_term,f_term,corners,source) 
##plt=demo_32.plotter()
##plt.show()
#
#demo_64=SFEM_2D(degree,'Data-selected/s_curve_mi_64_R_1.05.g',k_term,f_term,corners,source) 
#plt=demo_64.plotter(1)
#plt.show()


#R=125
#demo_2=SFEM_2D(degree,'Data-selected/s_curve_mi_2_R_1.25.g',k_term,f_term,corners,source) 
#plt = demo_2.plotter()
#plt.savefig('./figs/R_125_SFEM_1.pgf')
#plt.savefig('./figs/R_125_SFEM_1.png')
#plt.show()
#demo_4=SFEM_2D(degree,'Data-selected/s_curve_mi_4_R_1.25.g',k_term,f_term,corners,source) 
##demo_4.plotter()
##
#demo_8=SFEM_2D(degree,'Data-selected/s_curve_mi_8_R_1.25.g',k_term,f_term,corners,source) 
##demo_8.plotter()
##plt.show()
##
#demo_16=SFEM_2D(degree,'Data-selected/s_curve_mi_16_R_1.25.g',k_term,f_term,corners,source) 
##demo_16.plotter()
#
#demo_32=SFEM_2D(degree,'Data-selected/s_curve_mi_32_R_1.25.g',k_term,f_term,corners,source) 
##plt=demo_32.plotter()
##plt.show()
#
#demo_64=SFEM_2D(degree,'Data-selected/s_curve_mi_64_R_1.25.g',k_term,f_term,corners,source) 
##plt=demo_64.plotter(0)
##plt.show()

###
R=200
demo_2=SFEM_2D(degree,'Data-selected/s_curve_mi_2_R_2.g',k_term,f_term,corners,source) 
plt = demo_2.plotter()
plt.savefig('./figs/R_200_SFEM_1.pgf')
plt.savefig('./figs/R_200_SFEM_1.png')

demo_4=SFEM_2D(degree,'Data-selected/s_curve_mi_4_R_2.g',k_term,f_term,corners,source) 
#demo_4.plotter()
#
demo_8=SFEM_2D(degree,'Data-selected/s_curve_mi_8_R_2.g',k_term,f_term,corners,source) 
#demo_8.plotter()
#plt.show()
#
demo_16=SFEM_2D(degree,'Data-selected/s_curve_mi_16_R_2.g',k_term,f_term,corners,source) 
#demo_16.plotter()

demo_32=SFEM_2D(degree,'Data-selected/s_curve_mi_32_R_2.g',k_term,f_term,corners,source) 
plt=demo_32.plotter()
plt.show()

demo_64=SFEM_2D(degree,'Data-selected/s_curve_mi_64_R_2.g',k_term,f_term,corners,source) 
#plt=demo_64.plotter(0)
#plt.show()

"""
Data for IGA
"""
"""
Multiplicites
"""
#h refinment
#multiplicity_1=np.array([[1,1],[1,1],[1,2],[1,2],[1,1],[1,1]])

multiplicity_1=np.array([[1,1],[1,1],[1,2],[1,2],[1,2],[1,2],[1,1],[1,1]])
multiplicity_2=multiplicity_1
multiplicity_3=multiplicity_1
multiplicity_4=multiplicity_1
#multiplicity_5=np.array([[5,5],[5,5],[5,5],[5,5],[5,5],[5,5]])-4

multiplicity_5=np.array([[5,5],[5,5],[5,5],[5,5],[5,5],[5,5],[5,5],[5,5]])-4


#p refinement
multiplicity_6=multiplicity_1+1
multiplicity_7=multiplicity_1+2
multiplicity_8=multiplicity_1+3
multiplicity_9=multiplicity_1+4

#k refinement
#multiplicity_10=np.array([[1,1],[1,1],[1,1],[1,1],[1,1],[1,1]])

multiplicity_10=np.array([[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1]])+1
multiplicity_11=multiplicity_10
multiplicity_12=multiplicity_10
multiplicity_13=multiplicity_10

#mixed
#multiplicity_14=np.array([[1,1],[1,1],[1,1],[1,1],[1,1],[1,1]])

multiplicity_14=np.array([[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1]])
multiplicity_15=multiplicity_14
multiplicity_16=multiplicity_14
multiplicity_17=multiplicity_14

"""
Order
"""
#h refinement

#order_1=np.array([[1,1],[1,1],[1,2],[1,2],[1,1],[1,1]])

order_1=np.array([[1,1],[1,1],[1,2],[1,2],[1,2],[1,2],[1,1],[1,1]])
order_2=order_1
order_3=order_1
order_4=order_1
#order_5=np.array([[7,7],[7,7],[7,7],[7,7],[7,7],[7,7]])+0
order_5=np.array([[8,8],[8,8],[8,2],[8,2],[8,2],[8,2],[8,8],[8,8]])

#p refinement
order_6=order_1+1
order_7=order_1+2
order_8=order_1+3
order_9=order_1+4

#k refinement
order_10=order_1+2
order_11=order_1+3
order_12=order_1+4
order_13=order_1+5

#mixed
order_14=np.array([[2,2],[2,2],[2,2],[2,2],[2,2],[2,2],[2,2],[2,2]])
#order_14=np.array([[2,2],[2,2],[2,2],[2,2],[2,2],[2,2]])+1
order_15=np.array([[3,3],[3,3],[3,2],[3,2],[3,2],[3,2],[3,3],[3,3]])
order_16=np.array([[4,4],[4,4],[4,2],[4,2],[4,2],[4,2],[4,4],[4,4]])
order_17=np.array([[5,5],[5,5],[5,2],[5,2],[5,2],[5,2],[5,5],[5,5]])

"""
Number elements
"""
#h refinement
number_elements_1=np.array([[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1]])
number_elements_2=np.array([[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0]])+1

#number_elements_1=np.array([[1,1],[1,1],[1,2],[1,2],[1,1],[1,1]])
#number_elements_2=np.array([[2,2],[2,2],[2,2],[2,2],[2,2],[2,2]])

number_elements_3=number_elements_2*2
number_elements_4=number_elements_2*4
n=int(36)
number_elements_5=np.array([[n,n/2],[n,n/2],[n,3],[n,3],[n,3],[n,3],[n,n/2],[n,n/2]]).astype('int')

#p refinement
number_elements_6=number_elements_1
number_elements_7=number_elements_1
number_elements_8=number_elements_1
number_elements_9=number_elements_1

#hpk refinement
number_elements_10=number_elements_1
number_elements_11=number_elements_1
number_elements_12=number_elements_1
number_elements_13=number_elements_1


#mixed
number_elements_14=np.array([[2,2],[2,2],[2,1],[2,1],[2,1],[2,1],[2,2],[2,2]])
number_elements_15=np.array([[3,3],[3,3],[3,1],[3,1],[3,1],[3,1],[3,3],[3,3]])
number_elements_16=np.array([[4,4],[4,4],[4,1],[4,1],[4,1],[4,1],[4,4],[4,4]])
number_elements_17=number_elements_16



#Gives which previously defined patches the current patch shares a boundary with                                     
patch_connect_shared_patch=np.array([[1,0],[1,0],[2,3],[3,0],[4,5],[5,0],[6,7]])-1          
#Gives direction on which patches share boundary                      
patch_connect_xi_or_eta=np.array([[2,0],[1,0],[1,2],[1,0],[1,2],[1,0],[1,2]])-1 
#Gives coordinate of xi or eta at boundary with respect to sharing patch
patch_connect_bound_end=np.array([[2,0],[2,0],[2,2],[2,0],[2,2],[2,0],[2,2]])-1
             
             
##Gives which previously defined patches the current patch shares a boundary with                                     
#patch_connect_shared_patch=np.array([[1,0],[1,0],[2,3],[3,0],[4,5]])-1          
##Gives direction on which patches share boundary                      
#patch_connect_xi_or_eta=np.array([[2,0],[1,0],[1,2],[1,0],[1,2]])-1 
##Gives coordinate of xi or eta at boundary with respect to sharing patch
#patch_connect_bound_end=np.array([[2,0],[2,0],[2,2],[2,0],[2,2]])-1




patch_connect_info=np.dstack((patch_connect_shared_patch,patch_connect_xi_or_eta,patch_connect_bound_end))  


IGA_mesh_1=Ippre.patch_data(order_1,number_elements_1,multiplicity_1,patch_connect_info)   
IGA_mesh_2=Ippre.patch_data(order_2,number_elements_2,multiplicity_2,patch_connect_info)  
IGA_mesh_3=Ippre.patch_data(order_3,number_elements_3,multiplicity_3,patch_connect_info)  
IGA_mesh_4=Ippre.patch_data(order_4,number_elements_4,multiplicity_4,patch_connect_info)  
IGA_mesh_5=Ippre.patch_data(order_5,number_elements_5,multiplicity_5,patch_connect_info)  

IGA_mesh_6=Ippre.patch_data(order_6,number_elements_6,multiplicity_6,patch_connect_info)  
IGA_mesh_7=Ippre.patch_data(order_7,number_elements_7,multiplicity_7,patch_connect_info)  
IGA_mesh_8=Ippre.patch_data(order_8,number_elements_8,multiplicity_8,patch_connect_info)  
IGA_mesh_9=Ippre.patch_data(order_9,number_elements_9,multiplicity_9,patch_connect_info)  

IGA_mesh_10=Ippre.patch_data(order_10,number_elements_10,multiplicity_10,patch_connect_info)  
IGA_mesh_11=Ippre.patch_data(order_11,number_elements_11,multiplicity_11,patch_connect_info)  
IGA_mesh_12=Ippre.patch_data(order_12,number_elements_12,multiplicity_12,patch_connect_info)  
IGA_mesh_13=Ippre.patch_data(order_13,number_elements_13,multiplicity_13,patch_connect_info)  

IGA_mesh_14=Ippre.patch_data(order_14,number_elements_14,multiplicity_14,patch_connect_info)  
IGA_mesh_15=Ippre.patch_data(order_15,number_elements_15,multiplicity_15,patch_connect_info)  
IGA_mesh_16=Ippre.patch_data(order_16,number_elements_16,multiplicity_16,patch_connect_info)  
IGA_mesh_17=Ippre.patch_data(order_17,number_elements_17,multiplicity_17,patch_connect_info)  

      
#http://www.cs.mtu.edu/~shene/COURSES/cs3621/NOTES/spline/NURBS/RB-circles.html
      
    
#R=1.25   
#theta=math.acos(1-2/R**2) 
#B=np.array([[0,0,1] #Nodes numbered similar to global bases functions
#                                ,[4,0,1]
#                                ,[0,2,1]
#                                ,[4,2,1]
#                                
#                                ,[8,0,1]
#                                ,[8,2,1]
#                                
#                                ,[0,3,1]
#                                ,[4+math.tan(.5*theta),3,math.sin(np.pi/2-.5*theta)]
#                                ,[0,4,1]
#                                ,[4,4,1]
#                        
#                                ,[8,3,1]
#                                ,[8,4,1]
#                                                              
#                                
#                                ,[0,5,1]
#                                ,[4-math.tan(.5*theta),5,math.sin(np.pi/2-.5*theta)]
#                                ,[0,6,1]
#                                ,[4,6,1]
#                                                             
#                                ,[8,5,1]
#                                ,[8,6,1]
#                                                               
#                    
#                                ,[0,8,1]
#                                ,[4,8,1]
#                                
#                                ,[8,8,1]]
#                                ,dtype=float)

    
#R=1.05   
#theta=math.acos(1-2/R**2) 
#B=np.array([[0,0,1] #Nodes numbered similar to global bases functions
#                                ,[4,0,1]
#                                ,[0,2,1]
#                                ,[4,2,1]
#                                
#                                ,[8,0,1]
#                                ,[8,2,1]
#                                
#                                ,[0,3,1]
#                                ,[4+math.tan(.5*theta),3,math.sin(np.pi/2-.5*theta)]
#                                ,[0,4,1]
#                                ,[4,4,1]
#                        
#                                ,[8,3,1]
#                                ,[8,4,1]
#                                                              
#                                
#                                ,[0,5,1]
#                                ,[4-math.tan(.5*theta),5,math.sin(np.pi/2-.5*theta)]
#                                ,[0,6,1]
#                                ,[4,6,1]
#                                                             
#                                ,[8,5,1]
#                                ,[8,6,1]
#                                                               
#                    
#                                ,[0,8,1]
#                                ,[4,8,1]
#                                
#                                ,[8,8,1]]
#                                ,dtype=float)
#       


R=2   
theta=math.acos(1-2/R**2)  
#ROC=.8
B=np.array([[0,0,1] #Nodes numbered similar to global bases functions
                                ,[4,0,1]
                                ,[0,2,1]
                                ,[4,2,1]
                                
                                ,[8,0,1]
                                ,[8,2,1]
                                
                                ,[0,3,1]
                                ,[4+math.tan(.5*theta),3,math.sin(np.pi/2-.5*theta)]
                                ,[0,4,1]
                                ,[4,4,1]
                        
                                ,[8,3,1]
                                ,[8,4,1]
                                                              
                                
                                ,[0,5,1]
                                ,[4-math.tan(.5*theta),5,math.sin(np.pi/2-.5*theta)]
                                ,[0,6,1]
                                ,[4,6,1]
                                                             
                                ,[8,5,1]
                                ,[8,6,1]
                                                               
                    
                                ,[0,8,1]
                                ,[4,8,1]
                                
                                ,[8,8,1]]
                                ,dtype=float)
                                
                                
#Shift control point to have center at origin
B[:,0:2]=B[:,0:2]-4            
B[:,0:2]=B[:,0:2]*100
#Refine coarsest control net                    
control_net_1=B[:]
control_net_2=IGA_refine.refine_multipatch(IGA_mesh_1,IGA_mesh_2,B)
control_net_3=IGA_refine.refine_multipatch(IGA_mesh_1,IGA_mesh_3,B)
control_net_4=IGA_refine.refine_multipatch(IGA_mesh_1,IGA_mesh_4,B)
control_net_5=IGA_refine.refine_multipatch(IGA_mesh_1,IGA_mesh_5,B)
#
control_net_6=IGA_refine.refine_multipatch(IGA_mesh_1,IGA_mesh_6,B)
control_net_7=IGA_refine.refine_multipatch(IGA_mesh_1,IGA_mesh_7,B)
control_net_8=IGA_refine.refine_multipatch(IGA_mesh_1,IGA_mesh_8,B)
control_net_9=IGA_refine.refine_multipatch(IGA_mesh_1,IGA_mesh_9,B)
##
control_net_10=IGA_refine.refine_multipatch(IGA_mesh_1,IGA_mesh_10,B)
control_net_11=IGA_refine.refine_multipatch(IGA_mesh_1,IGA_mesh_11,B)
control_net_12=IGA_refine.refine_multipatch(IGA_mesh_1,IGA_mesh_12,B)
control_net_13=IGA_refine.refine_multipatch(IGA_mesh_1,IGA_mesh_13,B)
##
control_net_14=IGA_refine.refine_multipatch(IGA_mesh_1,IGA_mesh_14,B)
control_net_15=IGA_refine.refine_multipatch(IGA_mesh_1,IGA_mesh_15,B)
control_net_16=IGA_refine.refine_multipatch(IGA_mesh_1,IGA_mesh_16,B)
control_net_17=IGA_refine.refine_multipatch(IGA_mesh_1,IGA_mesh_17,B)    
      
#Physical paramters
k=lambda x,y: np.ones(len(x))*10.**-10
f=lambda x,y: np.ones(len(x))*0  

#nuemann_bound=np.array([[1,0,0],
#                        [0,0,0],
#                        [0,0,1],
#                        [1,1,1],
#                        [1,0,2],
#                        [1,1,3],
#                        [1,0,4],
#                        [1,1,5],
#                        [1,0,6],
#                        [0,1,6],
#                        [0,1,7],
#                        [1,1,7]])
       

#nuemann_bound=np.array([
#                        [0,0,0],
#                        [0,0,1],
#                        
#                       
#                        [0,1,4],
#                        [0,1,5]
#                        ])
           
nuemann_bound=np.array([[1,1,2],[1,1,4]])             
#flux=np.array([.0625,.125,.125,.0625,.0625,.0625,.0625,.0625,.0625,.125,.125,.0625])*10E-4
#flux=np.array([1,1,1,1])*10E-6
#flux=np.ones(12)*10E-4
#flux=np.ones(10)*10E-6
flux=np.ones(2)*-50E-7

#fhat=np.ones(9)*10
source_strength=-50E-7
#source_strength=1.

IGA_a_mesh_1=IGA_2D(order_1,number_elements_1,multiplicity_1,patch_connect_info,control_net_1,k,f,source_strength,nuemann_bound,flux)
IGA_a_mesh_2=IGA_2D(order_2,number_elements_2,multiplicity_2,patch_connect_info,control_net_2,k,f,source_strength,nuemann_bound,flux)
IGA_a_mesh_3=IGA_2D(order_3,number_elements_3,multiplicity_3,patch_connect_info,control_net_3,k,f,source_strength,nuemann_bound,flux)
IGA_a_mesh_4=IGA_2D(order_4,number_elements_4,multiplicity_4,patch_connect_info,control_net_4,k,f,source_strength,nuemann_bound,flux)
IGA_a_mesh_5=IGA_2D(order_5,number_elements_5,multiplicity_5,patch_connect_info,control_net_5,k,f,source_strength,nuemann_bound,flux)

IGA_a_mesh_6=IGA_2D(order_6,number_elements_6,multiplicity_6,patch_connect_info,control_net_6,k,f,source_strength,nuemann_bound,flux)
IGA_a_mesh_7=IGA_2D(order_7,number_elements_7,multiplicity_7,patch_connect_info,control_net_7,k,f,source_strength,nuemann_bound,flux)
IGA_a_mesh_8=IGA_2D(order_8,number_elements_8,multiplicity_8,patch_connect_info,control_net_8,k,f,source_strength,nuemann_bound,flux)
IGA_a_mesh_9=IGA_2D(order_9,number_elements_9,multiplicity_9,patch_connect_info,control_net_9,k,f,source_strength,nuemann_bound,flux)


#IGA_a_mesh_10=IGA_2D(order_10,number_elements_10,multiplicity_10,patch_connect_info,control_net_10,k,f,source_strength,nuemann_bound,flux)
#IGA_a_mesh_11=IGA_2D(order_11,number_elements_11,multiplicity_11,patch_connect_info,control_net_11,k,f,source_strength,nuemann_bound,flux)
#IGA_a_mesh_12=IGA_2D(order_12,number_elements_12,multiplicity_12,patch_connect_info,control_net_12,k,f,source_strength,nuemann_bound,flux)
#IGA_a_mesh_13=IGA_2D(order_13,number_elements_13,multiplicity_13,patch_connect_info,control_net_13,k,f,source_strength,nuemann_bound,flux)
#
IGA_a_mesh_14=IGA_2D(order_14,number_elements_14,multiplicity_14,patch_connect_info,control_net_14,k,f,source_strength,nuemann_bound,flux)
IGA_a_mesh_15=IGA_2D(order_15,number_elements_15,multiplicity_15,patch_connect_info,control_net_15,k,f,source_strength,nuemann_bound,flux)
IGA_a_mesh_16=IGA_2D(order_16,number_elements_16,multiplicity_16,patch_connect_info,control_net_16,k,f,source_strength,nuemann_bound,flux)
IGA_a_mesh_17=IGA_2D(order_17,number_elements_17,multiplicity_17,patch_connect_info,control_net_17,k,f,source_strength,nuemann_bound,flux)

#a=IGA_a_mesh_1.NURBS_bases(np.array([0]),np.array([.5]),3,None)
#d=IGA_a_mesh_1.Bspline_basis(3,np.array([.75]),np.array([ 0.  ,  0.  ,  0.  , 0,  0.5,  0.5 , 1.  ,  1.  ,  1. ,1. ]),3,1)
#b=IGA_a_mesh_1.patch_data.patch_global_2D(3).flatten()
#c=IGA_a_mesh_1.B[b]
#x=np.dot(a,c[:,0])
#y=np.dot(a,c[:,1])
#c[:,0:2]=c[:,0:2]/100
#c[:,1]=c[:,1]+2


#a=IGA_a_mesh_5.NURBS_bases(np.array([0]),np.array([.25]),3,None)
#d=IGA_a_mesh_5.Bspline_basis(3,np.array([.75]),np.array([ 0.  ,  0.  ,  0.  , 0,  0.5,  0.5 , 1.  ,  1.  ,  1. ,1. ]),3,1)
#b=IGA_a_mesh_5.patch_data.patch_global_2D(3).flatten()
#c=IGA_a_mesh_5.B[b]
#x=np.dot(a,c[:,0])
#y=np.dot(a,c[:,1])
#c[:,0:2]=c[:,0:2]/100
#c[:,1]=c[:,1]+2


plt = IGA_a_mesh_1.plotter(5,1,1)
plt.savefig('./figs/R_200_IGA_1.pgf')
plt.savefig('./figs/R_200_IGA_1.png')
plt.show()
#plt=IGA_a_mesh_2.plotter(5,1,1)
#plt.show
#plt=IGA_a_mesh_3.plotter(5,1,1)
#plt.show
#plt=IGA_a_mesh_4.plotter(5)
#plt.show
plt = IGA_a_mesh_5.plotter(5,0,0)
plt.savefig('./figs/R_200_ref_sol.pgf')
plt.savefig('./figs/R_200_ref_sol.png')


#plt=IGA_a_mesh_6.plotter(5)
#plt.show
#plt=IGA_a_mesh_7.plotter(5)
#plt.show
#plt=IGA_a_mesh_8.plotter(5)
#plt.show
#plt=IGA_a_mesh_9.plotter(5)
#plt.show
###
#plt=IGA_a_mesh_10.plotter(5)
#plt.show
#plt=IGA_a_mesh_11.plotter(5)
#plt.show
#plt=IGA_a_mesh_12.plotter(5)
#plt.show
#plt=IGA_a_mesh_13.plotter(5)
#plt.show

#plt=IGA_a_mesh_14.plotter(5)
#plt.show
#plt=IGA_a_mesh_15.plotter(5)
#plt.show
#plt=IGA_a_mesh_16.plotter(5)
#plt.show
"""
Creates vector of x,y points to be sampled in SFEM code
"""

##need griddata statement
#test_xi=np.linspace(.05,.95,20)
#test_eta=np.linspace(.05,.95,20)
#
#XI,ETA=np.meshgrid(test_xi,test_eta)
#test_xi=XI.ravel()
#test_eta=ETA.ravel()
#test_x=np.empty(0); test_y=np.empty(0)
#for pnum in np.arange(8):
#    R=IGA_a_mesh_1.NURBS_bases(test_xi,test_eta,pnum)
#    B_patch_ind=IGA_a_mesh_1.patch_data.patch_global_2D(pnum).flatten()
#    x_patch=IGA_a_mesh_1.B[B_patch_ind,0]
#    y_patch=IGA_a_mesh_1.B[B_patch_ind,1]
#    x_pt=np.sum(R*x_patch,axis=1)
#    y_pt=np.sum(R*y_patch,axis=1)
#    test_x=np.append(test_x,x_pt)
#    test_y=np.append(test_y,y_pt)


n=20
test_xi1=np.linspace(.01,.99,n); test_eta1=np.linspace(.01,.99,n)

test_xi2=np.linspace(.01,.99,n); test_eta2=np.linspace(.01,.99,n)

test_xi3=np.linspace(.01,.99,n); test_eta3=np.linspace(.01,.99,n)

test_xi4=np.linspace(.01,.99,n); test_eta4=np.linspace(.01,.99,n)

test_xi5=np.linspace(.01,.99,n); test_eta5=np.linspace(.01,.99,n)

test_xi6=np.linspace(.01,.99,n); test_eta6=np.linspace(.01,.99,n)


XI1,ETA1=np.meshgrid(test_xi1,test_eta1); test_xi1=XI1.ravel(); test_eta1=ETA1.ravel()
XI2,ETA2=np.meshgrid(test_xi2,test_eta2); test_xi2=XI2.ravel(); test_eta2=ETA2.ravel()
XI3,ETA3=np.meshgrid(test_xi3,test_eta3); test_xi3=XI3.ravel(); test_eta3=ETA3.ravel()
XI4,ETA4=np.meshgrid(test_xi4,test_eta4); test_xi4=XI4.ravel(); test_eta4=ETA4.ravel()
XI5,ETA5=np.meshgrid(test_xi5,test_eta5); test_xi5=XI5.ravel(); test_eta5=ETA5.ravel()
XI6,ETA6=np.meshgrid(test_xi6,test_eta6); test_xi6=XI6.ravel(); test_eta6=ETA6.ravel()

test_xi=np.empty(0); test_eta=np.empty(0); test_x=np.empty(0); test_y=np.empty(0); test_pnum=np.empty(0)

grp1=[0]; grp2=[1]; grp3=[2,4]; grp4=[3,5]; grp5=[6]; grp6=[7]
for pnum in np.arange(8):
    if pnum in grp1:
        R=IGA_a_mesh_1.NURBS_bases(test_xi1,test_eta1,pnum)
        B_patch_ind=IGA_a_mesh_1.patch_data.patch_global_2D(pnum).flatten()
        x_patch=IGA_a_mesh_1.B[B_patch_ind,0]
        y_patch=IGA_a_mesh_1.B[B_patch_ind,1]
        x_pt=np.sum(R*x_patch,axis=1)
        y_pt=np.sum(R*y_patch,axis=1)
        test_x=np.append(test_x,x_pt)
        test_y=np.append(test_y,y_pt)
        test_xi=np.append(test_xi,test_xi1)
        test_eta=np.append(test_eta,test_eta1)
        test_pnum=np.append(test_pnum,pnum)
    elif pnum in grp2:
        R=IGA_a_mesh_1.NURBS_bases(test_xi2,test_eta2,pnum)
        B_patch_ind=IGA_a_mesh_1.patch_data.patch_global_2D(pnum).flatten()
        x_patch=IGA_a_mesh_1.B[B_patch_ind,0]
        y_patch=IGA_a_mesh_1.B[B_patch_ind,1]
        x_pt=np.sum(R*x_patch,axis=1)
        y_pt=np.sum(R*y_patch,axis=1)
        test_x=np.append(test_x,x_pt)
        test_y=np.append(test_y,y_pt)
        test_xi=np.append(test_xi,test_xi2)
        test_eta=np.append(test_eta,test_eta2)
        test_pnum=np.append(test_pnum,pnum)
    elif pnum in grp3:
        R=IGA_a_mesh_1.NURBS_bases(test_xi3,test_eta3,pnum)
        B_patch_ind=IGA_a_mesh_1.patch_data.patch_global_2D(pnum).flatten()
        x_patch=IGA_a_mesh_1.B[B_patch_ind,0]
        y_patch=IGA_a_mesh_1.B[B_patch_ind,1]
        x_pt=np.sum(R*x_patch,axis=1)
        y_pt=np.sum(R*y_patch,axis=1)
        test_x=np.append(test_x,x_pt)
        test_y=np.append(test_y,y_pt)
        test_xi=np.append(test_xi,test_xi3)
        test_eta=np.append(test_eta,test_eta3)
        test_pnum=np.append(test_pnum,pnum)
    elif pnum in grp4:
        R=IGA_a_mesh_1.NURBS_bases(test_xi4,test_eta4,pnum)
        B_patch_ind=IGA_a_mesh_1.patch_data.patch_global_2D(pnum).flatten()
        x_patch=IGA_a_mesh_1.B[B_patch_ind,0]
        y_patch=IGA_a_mesh_1.B[B_patch_ind,1]
        x_pt=np.sum(R*x_patch,axis=1)
        y_pt=np.sum(R*y_patch,axis=1)
        test_x=np.append(test_x,x_pt)
        test_y=np.append(test_y,y_pt)
        test_xi=np.append(test_xi,test_xi4)
        test_eta=np.append(test_eta,test_eta4)
        test_pnum=np.append(test_pnum,pnum)
    elif pnum in grp5:
        R=IGA_a_mesh_1.NURBS_bases(test_xi5,test_eta5,pnum)
        B_patch_ind=IGA_a_mesh_1.patch_data.patch_global_2D(pnum).flatten()
        x_patch=IGA_a_mesh_1.B[B_patch_ind,0]
        y_patch=IGA_a_mesh_1.B[B_patch_ind,1]
        x_pt=np.sum(R*x_patch,axis=1)
        y_pt=np.sum(R*y_patch,axis=1)
        test_x=np.append(test_x,x_pt)
        test_y=np.append(test_y,y_pt)
        test_xi=np.append(test_xi,test_xi5)
        test_eta=np.append(test_eta,test_eta5)
        test_pnum=np.append(test_pnum,pnum)
    elif pnum in grp6:
        R=IGA_a_mesh_1.NURBS_bases(test_xi6,test_eta6,pnum)
        B_patch_ind=IGA_a_mesh_1.patch_data.patch_global_2D(pnum).flatten()
        x_patch=IGA_a_mesh_1.B[B_patch_ind,0]
        y_patch=IGA_a_mesh_1.B[B_patch_ind,1]
        x_pt=np.sum(R*x_patch,axis=1)
        y_pt=np.sum(R*y_patch,axis=1)
        test_x=np.append(test_x,x_pt)
        test_y=np.append(test_y,y_pt)
        test_xi=np.append(test_xi,test_xi6)
        test_eta=np.append(test_eta,test_eta6)
        test_pnum=np.append(test_pnum,pnum)
 


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
    

def L2_error_SFEM(object2,object1,test_x,test_y,test_xi,test_eta,n):
    """
    Finds L2 error between 2 meshes using node locations of finer mesh as test points. Object 1 gives the IGA solution, object 2 the SFEM solution
    """
    x=test_x
    y=test_y
    xi=test_xi
    eta=test_eta
#    x=object2.x
#    y=object2.y
    
#    for pnum in range(8):
    for pt_num in range(len(test_x)):
        pnum=(np.floor(pt_num/n**2)).astype('int')
        if pt_num==0 and pnum==0:
            ob_1_approx=object1.approx_function(xi[pt_num],eta[pt_num],pnum)
            ob_2_approx=object2.approx_function(x[pt_num],y[pt_num])
#                if ob_2_approx==-1:
#                    error=np.empty(0)
#                    continue
#                else:
            error=np.array([(ob_1_approx-ob_2_approx)**2])
             
        else:
            ob_1_approx=object1.approx_function(xi[pt_num],eta[pt_num],pnum)
            ob_2_approx=object2.approx_function(x[pt_num],y[pt_num])
#                if ob_2_approx==-1:
#                    continue
#                else:
            err=np.array([(ob_1_approx-ob_2_approx)**2])
            error=np.append(error,err)
            
    return np.sqrt(np.sum(error))


#    
def L2_error_SFEM_v2(object2,object1,test_x,test_y,n):
    """
    Finds L2 error between 2 meshes using node locations of finer mesh as test points. Object 1 gives the IGA solution, object 2 the SFEM solution
    """
    x=test_x
    y=test_y
#    xi=test_xi
#    eta=test_eta
#    x=object2.x
#    y=object2.y
    
#    for pnum in range(8):
    for pt_num in range(len(test_x)):
        pnum=(np.floor(pt_num/n**2)).astype('int')
        if pt_num==0 and pnum==0:
            
            ob_1_approx=object1.approx_function(x[pt_num],y[pt_num])
            ob_2_approx=object2.approx_function(x[pt_num],y[pt_num])
#                if ob_2_approx==-1:
#                    error=np.empty(0)
#                    continue
#                else:
            error=(ob_1_approx-ob_2_approx)**2
#            print(ob_1_approx,ob_2_approx,error)
        else:
#                return object1.approx_function(x[pt_num+n**2*pnum],y[pt_num+n**2*pnum])

            ob_1_approx=object1.approx_function(x[pt_num],y[pt_num])
            ob_2_approx=object2.approx_function(x[pt_num],y[pt_num])
#                if ob_2_approx==-1:
#                    continue
#                else:
            err=(ob_1_approx-ob_2_approx)**2
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




def L2_error_IGA_v2(object_2,object_1,test_x,test_y,test_xi,test_eta,n):
    """
    Finds L2 error between 2 meshes using node locations of finer mesh as test points. Object 2 assumed to give the better solution
    """
    x=test_x
    y=test_y
    xi=test_xi
    eta=test_eta
#    x=object2.x
#    y=object2.y
    
    for pt_num in range(len(test_x)):
        pnum=(np.floor(pt_num/n**2)).astype('int')

        if pt_num==0 and pnum==0:
            ob_1_approx=object_1.approx_function(xi[pt_num],eta[pt_num],pnum)
            ob_2_approx=object_2.approx_function(x[pt_num],y[pt_num])
#                if ob_2_approx==-1:
#                    error=np.empty(0)
#                    continue
#                else:
            error=np.array([(ob_1_approx-ob_2_approx)**2])
             
        else:
            ob_1_approx=object_1.approx_function(xi[pt_num],eta[pt_num],pnum)
            ob_2_approx=object_2.approx_function(x[pt_num],y[pt_num])
#                if ob_2_approx==-1:
#                    continue
#                else:
            err=np.array([(ob_1_approx-ob_2_approx)**2])
            error=np.append(error,err)
            
    return np.sqrt(np.sum(error))  

    
    
    
"""
Mean percent error using SFEM as ref
"""

#L2_1_FD=MPE_FD_v2(CMG_1,demo_64,test_x,test_y,test_xi,test_eta)
#L2_2_FD=MPE_FD_v2(CMG_2,demo_64,test_x,test_y,test_xi,test_eta)
#L2_3_FD=MPE_FD_v2(CMG_3,demo_64,test_x,test_y,test_xi,test_eta)
#L2_4_FD=MPE_FD_v2(CMG_4,demo_64,test_x,test_y,test_xi,test_eta)
#
#
#L2_1_SFEM=MPE_SFEM_v2(demo_64,demo_2,test_x,test_y)
#L2_2_SFEM=MPE_SFEM_v2(demo_64,demo_4,test_x,test_y)
#L2_3_SFEM=MPE_SFEM_v2(demo_64,demo_8,test_x,test_y)
#L2_4_SFEM=MPE_SFEM_v2(demo_64,demo_16,test_x,test_y)
#L2_5_SFEM=MPE_SFEM_v2(demo_64,demo_32,test_x,test_y)
#
#L2_1_IGA=MPE_IGA_v2(demo_64,IGA_a_mesh_1,test_x,test_y,test_xi,test_eta)
#L2_2_IGA=MPE_IGA_v2(demo_64,IGA_a_mesh_2,test_x,test_y,test_xi,test_eta)
#L2_3_IGA=MPE_IGA_v2(demo_64,IGA_a_mesh_3,test_x,test_y,test_xi,test_eta)
#L2_4_IGA=MPE_IGA_v2(demo_64,IGA_a_mesh_4,test_x,test_y,test_xi,test_eta)
#
#L2_6_IGA=MPE_IGA_v2(demo_64,IGA_a_mesh_6,test_x,test_y,test_xi,test_eta)
#L2_7_IGA=MPE_IGA_v2(demo_64,IGA_a_mesh_7,test_x,test_y,test_xi,test_eta)
#L2_8_IGA=MPE_IGA_v2(demo_64,IGA_a_mesh_8,test_x,test_y,test_xi,test_eta)
#L2_9_IGA=MPE_IGA_v2(demo_64,IGA_a_mesh_9,test_x,test_y,test_xi,test_eta)
#
#L2_10_IGA=MPE_IGA_v2(demo_64,IGA_a_mesh_10,test_x,test_y,test_xi,test_eta)
#L2_11_IGA=MPE_IGA_v2(demo_64,IGA_a_mesh_11,test_x,test_y,test_xi,test_eta)
#L2_12_IGA=MPE_IGA_v2(demo_64,IGA_a_mesh_12,test_x,test_y,test_xi,test_eta)
#L2_13_IGA=MPE_IGA_v2(demo_64,IGA_a_mesh_13,test_x,test_y,test_xi,test_eta)
#
#L2_14_IGA=MPE_IGA_v2(demo_64,IGA_a_mesh_14,test_x,test_y,test_xi,test_eta)
#L2_15_IGA=MPE_IGA_v2(demo_64,IGA_a_mesh_15,test_x,test_y,test_xi,test_eta)
#L2_16_IGA=MPE_IGA_v2(demo_64,IGA_a_mesh_16,test_x,test_y,test_xi,test_eta)
#L2_17_IGA=MPE_IGA_v2(demo_64,IGA_a_mesh_17,test_x,test_y,test_xi,test_eta)
"""
Mean percent error using IGA as ref
"""
#L2_1_FD=MPE_FD(CMG_1,IGA_a_mesh_5,test_x,test_y,test_xi,test_eta)
#L2_2_FD=MPE_FD(CMG_2,IGA_a_mesh_5,test_x,test_y,test_xi,test_eta)
#L2_3_FD=MPE_FD(CMG_3,IGA_a_mesh_5,test_x,test_y,test_xi,test_eta)
#L2_4_FD=MPE_FD(CMG_4,IGA_a_mesh_5,test_x,test_y,test_xi,test_eta)
#
#L2_1_SFEM=MPE_SFEM(demo_2,IGA_a_mesh_5,test_x,test_y,test_xi,test_eta)
#L2_2_SFEM=MPE_SFEM(demo_4,IGA_a_mesh_5,test_x,test_y,test_xi,test_eta)
#L2_3_SFEM=MPE_SFEM(demo_8,IGA_a_mesh_5,test_x,test_y,test_xi,test_eta)
#L2_4_SFEM=MPE_SFEM(demo_16,IGA_a_mesh_5,test_x,test_y,test_xi,test_eta)
#L2_5_SFEM=MPE_SFEM(demo_32,IGA_a_mesh_5,test_x,test_y,test_xi,test_eta)
#L2_6_SFEM=L2_error_SFEM_v2(demo_64,demo_64,test_x,test_y)
#
#L2_1_IGA=MPE_IGA(IGA_a_mesh_5,IGA_a_mesh_1,test_xi,test_eta)
#L2_2_IGA=MPE_IGA(IGA_a_mesh_5,IGA_a_mesh_2,test_xi,test_eta)
#L2_3_IGA=MPE_IGA(IGA_a_mesh_5,IGA_a_mesh_3,test_xi,test_eta)
#L2_4_IGA=MPE_IGA(IGA_a_mesh_5,IGA_a_mesh_4,test_xi,test_eta)
#
#L2_6_IGA=MPE_IGA(IGA_a_mesh_5,IGA_a_mesh_6,test_xi,test_eta)
#L2_7_IGA=MPE_IGA(IGA_a_mesh_5,IGA_a_mesh_7,test_xi,test_eta)
#L2_8_IGA=MPE_IGA(IGA_a_mesh_5,IGA_a_mesh_8,test_xi,test_eta)
#L2_9_IGA=MPE_IGA(IGA_a_mesh_5,IGA_a_mesh_9,test_xi,test_eta)

#L2_10_IGA=MPE_IGA(IGA_a_mesh_5,IGA_a_mesh_10,test_xi,test_eta)
#L2_11_IGA=MPE_IGA(IGA_a_mesh_5,IGA_a_mesh_11,test_xi,test_eta)
#L2_12_IGA=MPE_IGA(IGA_a_mesh_5,IGA_a_mesh_12,test_xi,test_eta)
#L2_13_IGA=MPE_IGA(IGA_a_mesh_5,IGA_a_mesh_13,test_xi,test_eta)
#
#L2_14_IGA=MPE_IGA(IGA_a_mesh_5,IGA_a_mesh_14,test_xi,test_eta)
#L2_15_IGA=MPE_IGA(IGA_a_mesh_5,IGA_a_mesh_15,test_xi,test_eta)
#L2_16_IGA=MPE_IGA(IGA_a_mesh_5,IGA_a_mesh_16,test_xi,test_eta)
#L2_17_IGA=MPE_IGA(IGA_a_mesh_5,IGA_a_mesh_17,test_xi,test_eta)
"""
L2 error using IGA as ref
"""
L2_1_FD=L2_error_FD(CMG_1,IGA_a_mesh_5,test_x,test_y,test_xi,test_eta,n)
L2_2_FD=L2_error_FD(CMG_2,IGA_a_mesh_5,test_x,test_y,test_xi,test_eta,n)
L2_3_FD=L2_error_FD(CMG_3,IGA_a_mesh_5,test_x,test_y,test_xi,test_eta,n)
L2_4_FD=L2_error_FD(CMG_4,IGA_a_mesh_5,test_x,test_y,test_xi,test_eta,n)
     
L2_1_SFEM=L2_error_SFEM(demo_4,IGA_a_mesh_5,test_x,test_y,test_xi,test_eta,n)
L2_2_SFEM=L2_error_SFEM(demo_8,IGA_a_mesh_5,test_x,test_y,test_xi,test_eta,n)
L2_3_SFEM=L2_error_SFEM(demo_16,IGA_a_mesh_5,test_x,test_y,test_xi,test_eta,n)
L2_4_SFEM=L2_error_SFEM(demo_32,IGA_a_mesh_5,test_x,test_y,test_xi,test_eta,n)
#L2_6_SFEM=L2_error_SFEM(demo_64,IGA_a_mesh_5,test_x,test_y,test_xi,test_eta,n)

L2_1_IGA=L2_error_IGA(IGA_a_mesh_5,IGA_a_mesh_1,test_xi,test_eta,n)
L2_2_IGA=L2_error_IGA(IGA_a_mesh_5,IGA_a_mesh_2,test_xi,test_eta,n)
L2_3_IGA=L2_error_IGA(IGA_a_mesh_5,IGA_a_mesh_3,test_xi,test_eta,n)
L2_4_IGA=L2_error_IGA(IGA_a_mesh_5,IGA_a_mesh_4,test_xi,test_eta,n)


L2_6_IGA=L2_error_IGA(IGA_a_mesh_5,IGA_a_mesh_6,test_xi,test_eta,n)
L2_7_IGA=L2_error_IGA(IGA_a_mesh_5,IGA_a_mesh_7,test_xi,test_eta,n)
L2_8_IGA=L2_error_IGA(IGA_a_mesh_5,IGA_a_mesh_8,test_xi,test_eta,n)
L2_9_IGA=L2_error_IGA(IGA_a_mesh_5,IGA_a_mesh_9,test_xi,test_eta,n)


#L2_10_IGA=L2_error_IGA(IGA_a_mesh_5,IGA_a_mesh_10,test_xi,test_eta)
#L2_11_IGA=L2_error_IGA(IGA_a_mesh_5,IGA_a_mesh_11,test_xi,test_eta)
#L2_12_IGA=L2_error_IGA(IGA_a_mesh_5,IGA_a_mesh_12,test_xi,test_eta)
#L2_13_IGA=L2_error_IGA(IGA_a_mesh_5,IGA_a_mesh_13,test_xi,test_eta)
#

L2_14_IGA=L2_error_IGA(IGA_a_mesh_5,IGA_a_mesh_14,test_xi,test_eta,n)
L2_15_IGA=L2_error_IGA(IGA_a_mesh_5,IGA_a_mesh_15,test_xi,test_eta,n)
L2_16_IGA=L2_error_IGA(IGA_a_mesh_5,IGA_a_mesh_16,test_xi,test_eta,n)
L2_17_IGA=L2_error_IGA(IGA_a_mesh_5,IGA_a_mesh_17,test_xi,test_eta,n)

#
dof_FD_1=len(CMG_1.x)
dof_FD_2=len(CMG_2.x)
dof_FD_3=len(CMG_3.x)
dof_FD_4=len(CMG_4.x)
#dof_FD_5=len(CMG_5.x)
#dof_FD_6=len(CMG_6.x)
#dof_FD_7=len(CMG_7.x)

#dof_SFEM_0=len(demo_1.x)

dof_SFEM_1=len(demo_2.x)
dof_SFEM_2=len(demo_4.x)
dof_SFEM_3=len(demo_8.x)
dof_SFEM_4=len(demo_16.x)
dof_SFEM_5=len(demo_32.x)
dof_SFEM_6=len(demo_64.x)

dof_IGA_1=len(IGA_a_mesh_1.F)
dof_IGA_2=len(IGA_a_mesh_2.F)
dof_IGA_3=len(IGA_a_mesh_3.F)
dof_IGA_4=len(IGA_a_mesh_4.F)

dof_IGA_6=len(IGA_a_mesh_6.F)
dof_IGA_7=len(IGA_a_mesh_7.F)
dof_IGA_8=len(IGA_a_mesh_8.F)
dof_IGA_9=len(IGA_a_mesh_9.F)

#dof_IGA_10=len(IGA_a_mesh_10.F)
#dof_IGA_11=len(IGA_a_mesh_11.F)
#dof_IGA_12=len(IGA_a_mesh_12.F)
#dof_IGA_13=len(IGA_a_mesh_13.F)
#
dof_IGA_14=len(IGA_a_mesh_14.F)
dof_IGA_15=len(IGA_a_mesh_15.F)
dof_IGA_16=len(IGA_a_mesh_16.F)
dof_IGA_17=len(IGA_a_mesh_17.F)




dof_h=np.array([dof_IGA_1,dof_IGA_2,dof_IGA_3,dof_IGA_4])
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




#for pnum in range(8):
#    for pt_num in range(400):
#        if pt_num==0 and pnum==0:
#            ob_1_approx=IGA_a_mesh_5.approx_function(test_xi[pt_num],test_eta[pt_num],pnum)
#            ref_sol=np.array([ob_1_approx])
#        else:
#            ob_1_approx=IGA_a_mesh_5.approx_function(test_xi[pt_num],test_eta[pt_num],pnum)
#            ref_sol=np.append(ref_sol,ob_1_approx)
