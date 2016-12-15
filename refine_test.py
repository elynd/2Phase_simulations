# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 15:06:58 2016

@author: eric
"""
import IGA_refine
import numpy as np
import copy
"""
Testing h refine along s curve on coarsest mesh
"""    
#B=np.array([[0,-2,1],[4./3,-1,.6],[0,0,1],[-4./3,1,.6],[0,2,1]])
#B=np.array([[0,-2,1],[4./3,-1,.6],[0,0,1],[-4./3,1,.6],[0,2,1]])
B=np.array([[0,0,1],[1,0,1./np.sqrt(2)],[1,1,1],[1,2,1./np.sqrt(2)],[0,2,1]])

#B=np.array([[0,0,1],[1,0,1/np.sqrt(2)],[1,1,1]])
#B1=B[:,:]
#
KV_old=IGA_refine.knot_vector(2,2,2)
KV_new=IGA_refine.knot_vector(4,2,2)
n=5
n_new=4
p=2
ob1=IGA_refine.IGA_h_refine(KV_old,KV_new,copy.copy(B),n,n_new,p)
#B=np.array([[0,0,1],[1,0,1/np.sqrt(2)],[1,1,1]])


#B1=ob1.B_new

"""
Testing p refine. Refined to C_0 continuity
"""
p=2
p_new=3
B_old=B
#KV_old=KV_new
ob2=IGA_refine.IGA_p_refine(p,p_new,copy.copy(B_old),KV_old)



#Checks out
"""
Testing knot removal
"""
p=3
KV_old2=IGA_refine.knot_vector(2,3,3)
B_old2=ob2.B_new
num=1
ob3=IGA_refine.IGA_knot_removal(p,KV_old2,copy.copy(B_old2),num)
print(ob3.B_new)
#KV_new_remove=IGA_knot_removal(p,KV_old,B_old,num).KV_new


"""
Testing single element
"""

"""
Testing h refine along s curve on coarsest mesh
"""    
#B=np.array([[0,-2,1],[1,-2,1/np.sqrt(2)],[1,-1,1],[1,0,1/np.sqrt(2)],[0,0,1],[-1,0,1/np.sqrt(2)],[-1,1,1],[-1,2,1/np.sqrt(2)],[0,2,1]])
##B=np.array([[0,-2,1],[1,-2,1/np.sqrt(2)],[1,-1,1],[1,0,1/np.sqrt(2)],[0,0,1]])
#
##B=np.array([[0,0,1],[0,1,1]])
#KV_old=knot_vector(2,2,2)
#KV_new=knot_vector(2,2,2)
#n=5
#n_new=0
#p=2
#B_new=IGA_h_refine(KV_old,KV_new,B,n,n_new,p).B_new

#plt.figure(4)
#plt.scatter(B_new[:,0],B_new[:,1],color='blue')
#plt.scatter(B[:,0],B[:,1],color='red')
#Checks out

"""
Testing p refine. Refined to C_0 continuity
"""
#p=2
#p_new=3
#KV_old=KV_new[:]
#B_old=ob1.B_old
#B_new_p=IGA_p_refine(p,p_new,B_old,KV_old).B_new
#KV_new_p=IGA_p_refine(p,p_new,B_old,KV_old).KV_new

#plt.figure(5)
#plt.scatter(B_new_p[:,0],B_new_p[:,1])

#Checks out
"""
Testing knot removal
"""
#p=4
#KV_old=KV_new_p
#B_old=B_new_p
#num=3
#a=IGA_knot_removal(p,KV_old,B_old,num)
#a1,a2,a3,a4,a5=a.reduce_order_KV()
#
#plt.figure(8)
#plt.scatter(a1[:,0],a1[:,1])
#aa,bb,cc=a.remove_knot(a3,a5,a4,a1,a2)
#b1,b2=a.reduce_order_KV()
#b1,b2=b.reduce_order_KV()
#b1,b2,b3,b4,b5=b.reduce_order_KV()

#plt.figure(7)
#plt.scatter(aa[:,0],aa[:,1])

"""
Testing surface refine
"""
#B=np.array([[-4,-2,1],[0,-2,1],[-4,-1.5,1],[1,-2,1/np.sqrt(2)],[-4,-1,1],[1,-1,1],\
#[-4,-.5,1],[1,0,1/np.sqrt(2)],[-4,0,1],[0,0,1],[-4,.5,1],[-1,0,1/np.sqrt(2)],\
#[-4,1,1],[-1,1,1],[-4,1.5,1],[-1,2,1/np.sqrt(2)],[-4,2,1],[0,2,1]])

#B=np.array([[0,2,1],
#            [4,2,1],
#            [0,3,1]
#            ,[6,2,1/np.sqrt(2)]
#            ,[0,4,1]
#            ,[6,4,1]
#            ,[0,5,1]
#            ,[6,6,1/np.sqrt(2)]
#            ,[0,6,1]
#            ,[4,6,1]])
#mult=np.array([[1,1],[2,1]])
#num_ele=np.array([[1,1],[4,4]])
#order=np.array([[1,1],[2,4]])

#a=Surface_refine(mult,num_ele,order,B)
#plt.scatter(a.B_new[:,0],a.B_new[:,1])