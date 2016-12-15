"""
IGA code
3/23/15
@author: Eric Lynd
"""
import numpy as np
from IGA_pre_processing import patch_data
import IGA_pre_processing as Ippre
from scipy.linalg import solve as solve
from pylab import *
import scipy.interpolate
import math
pgf_with_custom_preamble = {
    "font.family": "serif",   # use serif/main font for text elements
    "text.usetex": True,    # use inline math for ticks
    "pgf.rcfonts": False,   # don't setup fonts from rc parameters
}
matplotlib.rcParams.update(pgf_with_custom_preamble)
import matplotlib.pyplot as plt
import IGA_refine  
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix




class IGA_2D():
    
    """
    Solves 2D boundary value problem with line sources/sinks using IGA.
    """
    
    def __init__(self,order,number_elements,multiplicity,patch_connect_info,B,k,f,source_strength,nuemann_bound,flux):
            
        """
        INPUTS:
        
        order:               matrix of bases orders in xi and eta for each patch
        number_elements:     number of bases in xi and eta for eac patch
        mulitplicity:        uniform multiplicity of interna knots in xi and eta for each patch
        patch_connect_info:  Array describing along which common boundaries patches are found. See furthur description in IGA_pre_processing.py
        B:                   Control net
        k:                   Material modulus, as a function of x and y
        f:                   Forcing terms, as a function of x and y
        source_strength:     (Optional) The strength of the line source in the problem
        nuemann_bound:       Boundary data for nuemann condition
        flux:                Corresponding flux values for nuemann boundaries
        """

        """
        Initialization of terms. Will have to include natural condition inpputs in future versions
        """
        self.order=order
        self.number_elements=number_elements
        self.patch_connect_info=patch_connect_info
        self.B=B
        
        ###Modified ontrol net with extra degrees of freedom at well locations
        self.B_modified=np.vstack((self.B,self.B[0]))
        self.B_modified=np.vstack((self.B_modified,self.B[-1]))

        self.k=k
        self.f=f
        self.source_strength=source_strength
        self.patch_data=patch_data(order,number_elements,multiplicity,patch_connect_info)
        self.nuemann_bound=nuemann_bound
        self.flux=flux
        """
        Pre compute solution
        """
        self.create_solution()
        
    def parent_to_para(self,e,xi_tilde,eta_tilde,patch_num):
        """
        Maps from parent element to parametric space
        
        INPUTS:
        e:           element number (Python indeces)
        xi_tilde:    xi_tilde coordinate in parent space
        eta_tilde:   eta_tilde coordinate in parent space
        patch_num:   patch number (Python indeces)
        """
        
        """
        Retrieve pertinent connectivity arrays from pre processing object
        """
        IEN_patch=self.patch_data.local_IEN_array[patch_num]
        INC_patch=self.patch_data.INC_array[patch_num]
        
        """
        Retrieve pertinent knot vectors arrays from pre processing object
        """
        KV_xi_patch=self.patch_data.KV_xi_array[patch_num]
        KV_eta_patch=self.patch_data.KV_eta_array[patch_num]
        
        """
        Finds index of right/uppermost supporting basis of element. Essentially finds length of knot span to be used in mapping
        """
        if len(IEN_patch.shape)!=1:
            ni=INC_patch[0,IEN_patch[e,0]] #CHAnGED DEF OF e TO BE CONSISTENT WITH PYTHON. IF IEN_PATCH IS !D ARRAY CANT USE 2 INDECES
            nj=INC_patch[1,IEN_patch[e,0]]
        else:
            ni=INC_patch[0,IEN_patch[0]] #CHAnGED DEF OF e TO BE CONSISTENT WITH PYTHON. IF IEN_PATCH IS !D ARRAY CANT USE 2 INDECES
            nj=INC_patch[1,IEN_patch[0]]   
        
        """
        Calculates xi and eta coordinates (parameter space) based on knot span
        """
        xi=((KV_xi_patch[ni+1]-KV_xi_patch[ni])*xi_tilde+(KV_xi_patch[ni+1]+KV_xi_patch[ni]))/2.
        eta=((KV_eta_patch[nj+1]-KV_eta_patch[nj])*eta_tilde+(KV_eta_patch[nj+1]+KV_eta_patch[nj]))/2.
     
        return xi, eta
        
    def Bspline_basis0(self,xi,KV_patch):
        """
        Finds order zero bases, at lowest level of recursive Cox-de-Booor function. 
        
        INPUTS:
        
        xi:         xi or eta coordinate
        KV_patch:   Knot vector along which bases are found
        
        See https://github.com/johntfoster/IGA/blob/master/IGA.py for source.
        """     
#        a=np.where(np.all([KV_patch[:-1] <=  xi[:,None], 
#                                    xi[:,None] < KV_patch[1:]],axis=0), 1.0, 0.0)
        cond1 = np.array(KV_patch[:-1]) <=  xi[:, None]
        cond2 = xi[:, None] < np.array(KV_patch[1:]) 
        
        a=np.where(cond1 & cond2, 1.0, 0.0)
        
        index1=np.where(xi==1)[0]
        index0=np.where(KV_patch==0)[0]
        if len(index1)==0:
            return a

        else:
            a[index1,-len(index0)]=1
            return a
        
    
    def Bspline_basis(self,degree,xi,KV_patch,patch_num,xi_dir=0,compute_derivatives=False):
        """
        Recursive Cox-de-Boor function to compute basis functions and optionally their derivatives. 
        
        INPUTS:
        
        degree:               current degree of basis function, itearted from 0 through p during recursion
        xi:                   xi or eta coordinate
        KV_patch:             knot vector along which bases lie
        patch_num:            patch number (Python indeces)
        xi_dir:               xi direction (0) or eta direction (1)
        compute_derivatives:  Choose to output just basis function (False) or also derivatives (True)
        
        See https://github.com/johntfoster/IGA/blob/master/IGA.py for source.
        """
        
        """
        Recursively finds basis functions for decreasing basis function orders 
        """
        if degree == 0 :
            return self.Bspline_basis0(xi,KV_patch)
        else:
            basis_p_minus_1= self.Bspline_basis(degree-1,xi,KV_patch,patch_num,xi_dir)
        
        """
        Checks if direction is in xi or eta. Chooses p (xi degree) or q (eta degree) accordingly
        """
        if xi_dir==0:
            max_degree=self.order[patch_num,0]
        else:
            max_degree=self.order[patch_num,1]     
        
        """
        Assembles terms to be used in Bspline bases
        """
#        first_term_numerator = xi - KV_patch[:-degree] 
#        first_term_denominator = KV_patch[degree:] - KV_patch[:-degree]
#        
#        second_term_numerator = KV_patch[(degree + 1):] - xi
#        second_term_denominator = (KV_patch[(degree + 1):] - 
#                                  KV_patch[1:-degree])
#        
        first_term_numerator = xi[:, np.newaxis] - KV_patch[:-degree] 
        first_term_denominator = KV_patch[degree:] - KV_patch[:-degree]
        
        second_term_numerator = KV_patch[(degree + 1):] - xi[:, np.newaxis]
        second_term_denominator = (KV_patch[(degree + 1):] - 
                                  KV_patch[1:-degree])
        
        """
        Change numerator in last recursion if derivatives are desired
        """
        if compute_derivatives and degree == max_degree:
            
            first_term_numerator = np.ones((len(xi), 
                                            len(first_term_denominator))) * degree
            second_term_numerator = np.ones((len(xi), 
                                             len(second_term_denominator))) * -degree

            
        """
        Disable divide by zero error because we check for it
        """
        with np.errstate(divide='ignore', invalid='ignore'):
            first_term = np.where(first_term_denominator != 0.0, 
                                  (first_term_numerator / 
                                   first_term_denominator), 0.0)
            second_term = np.where(second_term_denominator != 0.0,
                                   (second_term_numerator / 
                                    second_term_denominator), 0.0)
                                    
        a=(first_term[:,:-1] * basis_p_minus_1[:,:-1] + second_term * basis_p_minus_1[:,1:])          
        
        return a
        ###Trying to account for strange result at xi=1
#        if xi==1 and compute_derivatives==0:
#            a[-1]=1
#        elif xi==1 and compute_derivatives==1:
#            return self.Bspline_basis(degree,.9999999999,KV_patch,patch_num,xi_dir,True)
#        else:
#            return a
        

     
    def NURBS_bases(self,xi,eta,patch_num,deriv=None,test=False):
        """
        Assembles numerators and denominators to complete definitions of functions with
        respect to parametric coordinates on a given patch. 
        
        INPUTS:
        
        xi:        xi coordinate
        eta:       eta coordinate
        patch_num: patch_number (Python indeces)
        deriv:     Options for outputs
                   1)Bases without derivatives (None)
                   2)Derivatives in xi only (dR_dxi)
                   3)Derivatives in eta only (dR_deta)
                   4)Derivatives in xi and eta (both)
                   5)Bases and all derivative (all)
        """
        
        """
        Since bases aren't defined at xi or eta=1, take the limit at xi or eta -> 1. 
        Perhaps there is a more elegant way to perform this?
        """
#        if xi==1:
#            return self.NURBS_bases(.999999999999,eta,patch_num,deriv)
#        if eta==1:
#            return self.NURBS_bases(xi,.9999999999999,patch_num,deriv)
#        
        """
        Find relevant weights from global basis function references on patch
        """
        control_point_ref=self.patch_data.patch_global_2D(patch_num).flatten()
        weights=self.B[control_point_ref,-1]
        
        """
        Finds knot vectors relevant to patch
        """
        KV_xi_patch=self.patch_data.KV_xi_array[patch_num]
        KV_eta_patch=self.patch_data.KV_eta_array[patch_num]
        
        """
        Finds patch basis order
        """
        p=self.order[patch_num,0]; q=self.order[patch_num,1]

        """
        Use previous Bspline basis function to get functions and their derivatives in xi and eta
        """
        N=self.Bspline_basis(p,xi,KV_xi_patch,patch_num,0)
        M=self.Bspline_basis(q,eta,KV_eta_patch,patch_num,1)
        """
        Initialize NURBS basis vector and matrix for NURBS basis derivatives
        """
        R=np.zeros((len(xi),len(N[0,:])*len(M[0,:])))
        """
        Broadcast weights over NURBS bases vectors
        """
        for row in range(len(xi)):
            R[row,:]=np.ravel(np.einsum('j,i',N[row,:],M[row,:]))*weights #ravel vs. flatten?        
        
        """
        Modify vectors to complete definition of NURBS bases   
        """
        sum_tot=np.sum(R,axis=1)
        for row in range(len(xi)):
            R[row,:]=R[row,:]/sum_tot[row]    
        
#        R=R/sum_tot
        
        """
        Branching statement that return different outputs based on deriv
        """
        if test==True:
            return N,M,R
                
        if deriv==None:
            return R
        else:
        
            dN_dxi=self.Bspline_basis(p,xi,KV_xi_patch,patch_num,0,True)
            dM_deta=self.Bspline_basis(q,eta,KV_eta_patch,patch_num,1,True)
            
            dR_dxi=np.zeros((len(R[0,:]),2,len(xi))) 
    
            for lvl in range(len(xi)):
                dR_dxi[:,0,lvl]=np.ravel(np.einsum('j,i',dN_dxi[lvl,:],M[lvl,:]))*weights    
                dR_dxi[:,1,lvl]=np.ravel(np.einsum('j,i',N[lvl,:],dM_deta[lvl,:]))*weights  
            
                sum_xi=np.sum(dR_dxi[:,0,lvl])
                sum_eta=np.sum(dR_dxi[:,1,lvl])
                
                dR_dxi[:,0,lvl]=(dR_dxi[:,0,lvl]*sum_tot[lvl]-R[lvl,:]*sum_xi)/sum_tot[lvl]**2
                dR_dxi[:,1,lvl]=(dR_dxi[:,1,lvl]*sum_tot[lvl]-R[lvl,:]*sum_eta)/sum_tot[lvl]**2  
            
            
            
                
            if deriv=='dR_dxi':
                return  dR_dxi[:,0,:]
            elif deriv=='dR_deta':
                return dR_dxi[:,1,:]
            elif deriv=='both':
                return dR_dxi
            elif deriv=='all':
                return R,dR_dxi
            
                
    def NURBS_bases_interface(self,eta,deriv=None):    
        """
        Computes NURBS bases along interface
        
        INPUTS:
        
        eta:       eta coordinate
        deriv:     Options for outputs
                   1)Bases without derivatives (None)
                   2)Derivatives in xi only (dR_dxi)
                   3)Derivatives in eta only (dR_deta)
                   4)Derivatives in xi and eta (both)
                   5)Bases and all derivative (all)
        """
        
        """
        Since bases aren't defined at xi or eta=1, take the limit at xi or eta -> 1. 
        Perhaps there is a more elegant way to perform this?
        """
#        if xi==1:
#            return self.NURBS_bases(.999999999999,eta,patch_num,deriv)
#        if eta==1:
#            return self.NURBS_bases(xi,.9999999999999,patch_num,deriv)
#        
        """
        Find relevant weights from global basis function references on patch
        """
        control_point_ref=self.patch_data.patch_global_2D(2)[:,-1]
        weights=self.B[control_point_ref,-1]
        
        """
        Finds knot vectors relevant to patch
        """
        KV_eta_patch=self.patch_data.KV_eta_array[2]
        
        """
        Finds patch basis order
        """
        q=self.order[2,1]

        """
        Use previous Bspline basis function to get functions and their derivatives in xi and eta
        """
        M=self.Bspline_basis(q,eta,KV_eta_patch,2,1)
  
        """
        Initialize NURBS basis vector and matrix for NURBS basis derivatives
        """
        R=np.zeros(len(M)) 
      
        """
        Broadcast weights over NURBS bases vectors
        """
        R=M*weights #ravel vs. flatten?        
        
        """
        Modify vectors to complete definition of NURBS bases   
        """
        sum_tot=np.sum(R)
                
        R=R/sum_tot
        
        """
        Branching statement that return different outputs based on deriv
        """
        if deriv==None:
            return R
        else:
        
            dM_deta=self.Bspline_basis(q,eta,KV_eta_patch,2,1,True)
    
            dR_deta=np.zeros(len(R)) 
    
            dR_deta[:]=dM_deta*weights    
          
            sum_eta=np.sum(dR_deta)
            
            dR_deta[:]=(dR_deta*sum_tot-R*sum_eta)/sum_tot**2
            
            if deriv=='dR_deta':
                return dR_deta
            elif deriv=='all':
                return R,dR_deta   
                
    def mappings(self,e,xi_tilde,eta_tilde,patch_num,test=False):
        """
        Finds Jacobian for mapping from parent to physical space. 
        
        INPUTS:
        
        e:          element number (Python indeces)
        xi_tilde:   xi_tilde coordinate in parent space
        eta_tilde:   eta_tilde coordinate in parent space
        patch_num:  patch_number (Python indeces)
        """
        
        """
        Map from parent to parameter space
        """
        xi,eta=self.parent_to_para(e,xi_tilde,eta_tilde,patch_num)
        
        """
        Find releveant connectvity arrays for patch
        """
        IEN_patch=self.patch_data.local_IEN_array[patch_num]
        INC_patch=self.patch_data.INC_array[patch_num]
        
        """
        Finds knot vectors relevant to patch
        """
        KV_xi_patch=self.patch_data.KV_xi_array[patch_num]
        KV_eta_patch=self.patch_data.KV_eta_array[patch_num]
        
        """
        Control net over patch
        """
        control_point_ref=self.patch_data.patch_global_2D(patch_num).flatten()
        B_patch=self.B[control_point_ref]
        
        """
        Finds knot location on basis that can be used for parent -> parpameter mapping
        """
        if len(IEN_patch.shape)!=1:
            ni=INC_patch[0,IEN_patch[e,0]] #bottom left corner of element
            nj=INC_patch[1,IEN_patch[e,0]]
        else:
            ni=INC_patch[0,IEN_patch[0]] #bottom left corner of element
            nj=INC_patch[1,IEN_patch[0]]
        
        """
        Calculate NURBS bases
        """
        R,dR_dxi=self.NURBS_bases(xi,eta,patch_num,'all')

        """
        Mapping from parameter to physical space
        """
        dx_dxi=np.zeros((2,2,len(xi))) #for 2D
        dxi_dx=np.zeros((2,2,len(xi))) #for 2D
        dR_dx=np.zeros((len(R[0,:]),2,len(xi)))
#        return dR_dx
        for lvl in range(len(xi)):
            dx_dxi[0,0,lvl]=np.sum(B_patch[:,0]*dR_dxi[:,0,lvl])
            dx_dxi[0,1,lvl]=np.sum(B_patch[:,0]*dR_dxi[:,1,lvl])    
            dx_dxi[1,0,lvl]=np.sum(B_patch[:,1]*dR_dxi[:,0,lvl])    
            dx_dxi[1,1,lvl]=np.sum(B_patch[:,1]*dR_dxi[:,1,lvl])   
            
#        np.einsum('ij,ijk->ijk',)
        """
        Compute inverse of gradient
        """
        det_dx_dxi=dx_dxi[0,0,:]*dx_dxi[1,1,:]-dx_dxi[0,1,:]*dx_dxi[1,0,:]
        
        dxi_dx[0,0,:]=dx_dxi[1,1,:]
        dxi_dx[1,0,:]=-dx_dxi[1,0,:]
        dxi_dx[0,1,:]=-dx_dxi[0,1,:]
        dxi_dx[1,1,:]=dx_dxi[0,0,:]
        
        for lvl in range(len(xi)):
            dxi_dx [:,:,lvl]=dxi_dx [:,:,lvl]/det_dx_dxi[lvl]   
        
        """
        Compute derivatives of basis functions with respect to physical coordinates     
        """
#        print(dR_dx,dxi_dx)
        for lvl in range(len(xi)):
            dR_dx[:,0,lvl]=dR_dxi[:,0,lvl]*dxi_dx[0,0,lvl]+dR_dxi[:,1,lvl]*dxi_dx[1,0,lvl] 
            dR_dx[:,1,lvl]=dR_dxi[:,0,lvl]*dxi_dx[0,1,lvl]+dR_dxi[:,1,lvl]*dxi_dx[1,1,lvl] 
        
        """
        Gradient of mapping from parent element to parameter space
        """
        dxi_dtildexi=np.zeros((2,2)) #2x2 for 2D
        J=np.zeros([2,2,len(xi)]) #Jacobian matrix
        detJ=0
        
        if len(IEN_patch.shape)!=1:
            ni=INC_patch[0,IEN_patch[e,0]] #bottom left corner of element
            nj=INC_patch[1,IEN_patch[e,0]]
        else:
            ni=INC_patch[0,IEN_patch[0]] #bottom left corner of element
            nj=INC_patch[1,IEN_patch[0]]
        
#        return ni,nj
        dxi_dtildexi[0,0]=(KV_xi_patch[ni+1]-KV_xi_patch[ni])/2. #check mapping
        dxi_dtildexi[1,1]=(KV_eta_patch[nj+1]-KV_eta_patch[nj])/2.
        
        """
        Gradient of mapping from parent to physical space  
        """
        for lvl in range(len(xi)):

            J[0,0,lvl]=np.sum(dx_dxi[0,:,lvl]*dxi_dtildexi[:,0])
            J[0,1,lvl]=np.sum(dx_dxi[0,:,lvl]*dxi_dtildexi[:,1])
            J[1,0,lvl]=np.sum(dx_dxi[1,:,lvl]*dxi_dtildexi[:,0])
            J[1,1,lvl]=np.sum(dx_dxi[1,:,lvl]*dxi_dtildexi[:,1])
        
        detJ=J[0,0,:]*J[1,1,:]-J[0,1,:]*J[1,0,:] 
        
        #Return same output as NURBS_bases to save computation time in assembly
        if test==False:
            return detJ,R,dR_dx
        else:
            return dx_dxi,dR_dxi
            
    def edge_integral(self,direction,boundary,patch,flux,test=False):
        """
        direction=0 for xi, 1 for eta
        boundary=0 for left/bottom, 1 for right/top
        """
        if direction==0:
            if boundary==0:
                global_indeces=self.patch_data.patch_global_2D(patch)[0,:]
                local_indeces=self.patch_data.local_IEN(patch,1)[0,:]
            elif boundary==1:
                global_indeces=self.patch_data.patch_global_2D(patch)[-1,:]
                local_indeces=self.patch_data.local_IEN(patch,1)[-1,:]

        else:
            if boundary==0:
                global_indeces=self.patch_data.patch_global_2D(patch)[:,0]
                local_indeces=self.patch_data.local_IEN(patch,1)[:,0]
            elif boundary==1:
                global_indeces=self.patch_data.patch_global_2D(patch)[:,-1]
                local_indeces=self.patch_data.local_IEN(patch,1)[:,-1]
                
#        print(global_indeces,local_indeces)
             
        pts,wts=np.polynomial.legendre.leggauss(np.max(self.order)+1)   
#        print(pts)
      
        for ele_num in range(self.number_elements[patch,direction]):
            if direction==0 and boundary==0:
                ele=ele_num
            elif direction==0 and boundary==1:
                max_ele=np.prod(self.number_elements[patch,:])-1
                ele=max_ele-self.number_elements[patch,direction]+ele_num+1
            elif direction==1 and boundary==0:
                ele=(ele_num)* self.number_elements[patch,0] 
            elif direction==1 and boundary==1:
                ele=(ele_num+1)* self.number_elements[patch,0]-1  
                
#            for int_pt in range(len(pts)):
#                     
##                print(ele,self.number_elements[patch,direction])
#                
#                if direction==0:
#                    if boundary==0:
#                        xi,eta=self.parent_to_para(ele,pts[int_pt],-1,patch)
#                    elif boundary==1:
#                        xi,eta=self.parent_to_para(ele,pts[int_pt],1,patch)
#                elif direction==1:
#                    if boundary==0:
#                        xi,eta=self.parent_to_para(ele,-1,pts[int_pt],patch)
#                    elif boundary==1:
#                        xi,eta=self.parent_to_para(ele,1,pts[int_pt],patch)
                
                     
#                print(ele,self.number_elements[patch,direction])
            filler=np.ones(len(pts))*1
            if direction==0:
                if boundary==0:
                    xi,eta=self.parent_to_para(ele,pts,-filler,patch)
                elif boundary==1:
                    xi,eta=self.parent_to_para(ele,pts,filler,patch)
            elif direction==1:
                if boundary==0:
                    xi,eta=self.parent_to_para(ele,-filler,pts,patch)
                elif boundary==1:
                    xi,eta=self.parent_to_para(ele,filler,pts,patch)
            """
            Finding Jacobians
            """
            R,dR_edge=self.NURBS_bases(xi,eta,patch,'all')
            
            edge_control_points=self.B[global_indeces]
            
            dR_edge=dR_edge[local_indeces,direction,:]
            R=R[:,local_indeces]
            
#            xpts=self.B[global_indeces,0]
#            fflux=lambda x: .1*np.sin(np.pi*x/800)
            fflux=lambda x: flux
#            flux=fflux(xpts)
            xi_locs,eta_locs=self.parent_to_para(ele_num,pts,np.ones(len(pts)),0)
            x_locs=xi_locs*800
            for lvl in range(len(pts)):
                dx_d=np.sum(dR_edge[:,lvl]*edge_control_points[:,0])
                dy_d=np.sum(dR_edge[:,lvl]*edge_control_points[:,1])
        
                detJ=np.sqrt(dx_d**2+dy_d**2)*(1./self.number_elements[patch,direction]/2.)
    #                print(detJ)
    #                if ele_num==3:
    #                    print(integrals,R)
                self.F[global_indeces]+=fflux(x_locs[lvl])*R[lvl,:]*detJ*wts[lvl]
                
        if test==True:
            return detJ,self.F,R,dR_edge
#        return self.F[global_indeces]
#        control_point_load=np.array([22,20,18,16,14,12,10,8,4])-1 #should be basis function load
#        fhat=np.ones(9)*10
#        self.F[control_point_load]+=fhat  
                
    def mapping_interface(self,eta,test=False):
        """
        Returns detJ for application of line source terms
        """

#        R,dR_deta_interface=self.NURBS_bases_interface(eta,'all')
#        R1,dR_deta_interface1=self.NURBS_bases_interface(1/self.number_elements[2,1]/2,'all')
        R,dR_deta_interface=self.NURBS_bases(1,eta,2,'all')

        
        cp_indeces=self.patch_data.patch_global_2D(2)[:,-1]
        
        interface_control_points=self.B[cp_indeces]
        loc_pts=self.patch_data.local_IEN(2,1)[:,-1]
        dR_deta_interface=dR_deta_interface[loc_pts,1]
        R=R[loc_pts]
        dx_deta=np.sum(dR_deta_interface*interface_control_points[:,0])
        dy_deta=np.sum(dR_deta_interface*interface_control_points[:,1])
        
#        return np.sqrt(dx_deta**2+dy_deta**2)
        if test==0:
            return np.sqrt(dx_deta**2+dy_deta**2)*(1./self.number_elements[2,1]/2.),R
#            return 2*np.pi/self.number_elements[2,1]/2,R

        else:
            return np.sqrt(dx_deta**2+dy_deta**2),(1./self.number_elements[2,1]/2.)
    
#    def mapping_interface(self,direction,boundary,patch,)
    def assemble(self,test=False):
        """
        Builds global stiffness matrix and force vector. 
        """
        f_term=self.f
        k_term=self.k
        nelem=self.number_elements
        
            
        if np.array_equal(self.patch_data.order[1],np.ones(2)*-1)==0:
            number_patches=len(nelem)
        else:
            number_patches=1
    
        number_global_bases=len(self.B) #number global basis functions
        
        """
        Integration points. May need to add additional points for log funcitons
        """
        eta_loc,xi_loc,wl=Ippre.compute_gauss_points_and_weights(np.max(self.order)+1,2) 

        """
        Iniitializing enriched matrices, as indicated by addition of terms to number of standard bases
        """
        self.K_en=np.zeros((number_global_bases+2,number_global_bases+2))
        self.F_en=np.zeros(number_global_bases+2)
        
        """
        Initializing un-enriched matriices for comparison
        """
        self.K=np.zeros((number_global_bases,number_global_bases))
        self.F=np.zeros(number_global_bases)
        
        #Looping through patches
        for aa in range(number_patches):
            
            #Find patch data and control net specific to each patch
            if number_patches!=1:
                Global_IEN_patch=self.patch_data.Global_IEN[self.patch_data.Global_IEN_length[aa,0]:self.patch_data.Global_IEN_length[aa,1]+1,:]
            else:
                Global_IEN_patch=self.patch_data.Global_IEN
                
            IEN_patch=self.patch_data.local_IEN_array[aa]
            control_point_ref=self.patch_data.patch_global_2D(aa).flatten()
            B_patch=self.B[control_point_ref]
           
            #Loop over elements in patch
            for bb in range(np.prod(nelem[aa])): 
                
                #Finds basis functions supported within element
                if len(Global_IEN_patch.shape)!=1:
                    global_basis_functions=Global_IEN_patch[bb,:]
                else:
                    global_basis_functions=Global_IEN_patch

                global_basis_functions=global_basis_functions[0:np.prod(self.order[aa]+1)]
                
                
                idx_grid = np.ix_(global_basis_functions, global_basis_functions)
                
#
#                """
#                Additional code for enrichment
#                """
#                if bb==0 and np.prod(self.number_elements[0])==1:
#                    global_basis_functions_en=range(np.prod(self.order[aa]+1)+2)
#                    
#                elif bb==range(np.prod(nelem[aa]))[-1]:
#                    global_basis_functions_en=np.hstack((global_basis_functions,len(self.B)+1)) #Adds last basis function
#                
#                elif bb==0:
#                    global_basis_functions_en=np.hstack((global_basis_functions,len(self.B))) #Adds second to last basis function
#                
#                else:
#                    global_basis_functions_en=global_basis_functions[:]
#                    
#                idx_grid_en=np.ix_(global_basis_functions_en, global_basis_functions_en)
#                
#                
#                return idx_grid_en


                """
                Solves mapping and NURBS bases at various integration points. Puts results into 2 or 3D array
                """
                   
                """
                Mappings to x,y space
                """
                detJ,R,dR_dx=self.mappings(bb,xi_loc,eta_loc,aa)
                
                """
                x and y for xi and eta on element bb in patch aa
                """
#                xfun=np.zeros(len(xi_loc));  yfun=np.zeros(len(xi_loc))
#                print(R,B_patch,xfun)
#                for lvl in range(len(xi_loc)):
#                    xfun[lvl]=np.sum(R[lvl,:]*B_patch[:,0])
#                    yfun[lvl]=np.sum(R[lvl,:]*B_patch[:,1])
                
                xfun=np.einsum('ij,j->i',R,B_patch[:,0])
                yfun=np.einsum('ij,j->i',R,B_patch[:,1])
                """
                Material modulus and forcing terms
                """
#                k=np.zeros(len(xi_loc));  f=np.zeros(len(xi_loc))
#                for lvl in range(len(xi_loc)):
#                    k[lvl]=k_term(xfun[lvl],yfun[lvl])
#                    f[lvl]=f_term(xfun[lvl],yfun[lvl])
                k=k_term(xfun,yfun)
                f=f_term(xfun,yfun)
                
                """
                Selecting NURBS bases and their derivatives supported on element
                """
                R=R[:,IEN_patch[bb,:]] #in reverse order now. IEN_patch may be single row
                dR_dx=dR_dx[IEN_patch[bb,:],:,:]
                
#                    R=R[::-1]
#                    dR_dx=np.flipud(dR_dx)
                """
                Making grid for both non enriched and enriched meshes
                """
                dRdx_grid=np.zeros((len(dR_dx[:,0,0]),len(dR_dx[:,0,0]),len(xi_loc)))
                dRdy_grid=np.zeros((len(dR_dx[:,0,0]),len(dR_dx[:,0,0]),len(xi_loc)))
             
#                for lvl in range(len(xi_loc)):
#
#                    dRdx_grid[:,:,lvl]=np.einsum('i,j',dR_dx[:,0,lvl],dR_dx[:,0,lvl])
#                    dRdy_grid[:,:,lvl]=np.einsum('i,j',dR_dx[:,1,lvl],dR_dx[:,1,lvl])
                dRdx_grid=np.einsum('ik,jk->ijk',dR_dx[:,0,:],dR_dx[:,0,:])     
                dRdy_grid=np.einsum('ik,jk->ijk',dR_dx[:,1,:],dR_dx[:,1,:]) 
                """
                Calculating pertinent entries in K and F
                """
#                    return  dRdy_grid_modified.shape,dRdy_grid_modified.shape,idx_grid_en
#                for lvl in range(len(xi_loc)):
#                    self.K[idx_grid]+=k[lvl]*(dRdx_grid[:,:,lvl]+dRdy_grid[:,:,lvl])*detJ[lvl]*wl[lvl]
#                    self.F[global_basis_functions]+=f[lvl]*R[lvl,:]*detJ[lvl]*wl[lvl]
                self.K[idx_grid]+=np.einsum('ijk,k->ij',(dRdx_grid[:,:,:]+dRdy_grid[:,:,:]),k*detJ*wl)
                self.F[global_basis_functions]+=np.einsum('ji,j->i',R,f*detJ*wl)
                
                if test==True:
                    return self.K,detJ
#                for cc in range(len(wl)):
#                   
#                    """
#                    Mappings to x,y space
#                    """
#                    detJ,R,dR_dx=self.mappings(bb,xi_loc[cc],eta_loc[cc],aa)
#                    
#                    """
#                    x and y for xi and eta on element bb in patch aa
#                    """
#                    xfun=np.sum(R*B_patch[:,0])
#                    yfun=np.sum(R*B_patch[:,1])
#                    
#                    """
#                    Material modulus and forcing terms
#                    """
#                    k=k_term(xfun,yfun)
#                    f=f_term(xfun,yfun)
#                    
#                    """
#                    Selecting NURBS bases and their derivatives supported on element
#                    """
#                    R=R[IEN_patch[bb,:]] #in reverse order now. IEN_patch may be single row
#                    dR_dx=dR_dx[IEN_patch[bb,:],:]
#                    
##                    R=R[::-1]
##                    dR_dx=np.flipud(dR_dx)
#                    """
#                    Making grid for both non enriched and enriched meshes
#                    """
#                    dRdx_grid=np.einsum('i,j',dR_dx[:,0],dR_dx[:,0])
#                    dRdy_grid=np.einsum('i,j',dR_dx[:,1],dR_dx[:,1])
#                                        
#                    """
#                    Calculating pertinent entries in K and F
#                    """
##                    return  dRdy_grid_modified.shape,dRdy_grid_modified.shape,idx_grid_en
#                    self.K[idx_grid]+=k*(dRdx_grid+dRdy_grid)*detJ*wl[cc]
#                    self.F[global_basis_functions]+=f*R*detJ*wl[cc]
                    

                    
                    
                    
#        return K,F  
      
    def boundary_conditions(self):
        """
        Applies essential boundary conditions using Lagrange multiplier method.
        ACurrently only applicable to model problem
        
        Can be simplified using bc's only at interpolatory corners
        """
        global_indeces_1=self.patch_data.patch_global_2D(0)
        global_indeces_2=self.patch_data.patch_global_2D(1)
        global_indeces_3=self.patch_data.patch_global_2D(2)
        global_indeces_4=self.patch_data.patch_global_2D(3)
        global_indeces_5=self.patch_data.patch_global_2D(4)
        global_indeces_6=self.patch_data.patch_global_2D(5)
        global_indeces_7=self.patch_data.patch_global_2D(6)
        global_indeces_8=self.patch_data.patch_global_2D(7)
      
#        global_indeces=self.patch_data.patch_global_2D(self.ess_data[0],self.ess_data[1])
        """       
        location of corners
        """
        br_corner=global_indeces_2[0,-1]
        bl_corner=global_indeces_1[0,0]
        tl_corner=global_indeces_7[-1,0]
        tr_corner=global_indeces_8[-1,-1]
        tm_point=global_indeces_8[-1,0]
        bm_point=global_indeces_1[0,-1]
        lm_point=global_indeces_3[-1,0]
        rm_point=global_indeces_4[-1,-1]
        
#        
#        boundary_pts_nat=np.hstack((bl_corner,tr_corner))        
#        flux_corners=np.array([1,-1])*10E-6
        
#        boundary_pts_nat=np.hstack((bl_corner,br_corner,tr_corner,tl_corner))        
#        flux_corners=np.array([1,1,1,1])*-10E-6
        """
        Laplace verification
        """
#        self.boundary_pts_ess=np.unique(np.hstack((global_indeces_1[0,:],global_indeces_1[:,0],global_indeces_1[:,-1])))  
#        boundary_pts_ess=self.boundary_pts_ess
#        u_hat_corners=np.zeros(len(boundary_pts_ess))
        
        """
        5 spot
        """
#        self.boundary_pts_ess=np.hstack((br_corner,tl_corner))     
#        boundary_pts_ess=np.hstack((br_corner,tl_corner))        
#        u_hat_corners=np.array([1,1])*0

        """
        Standard problem
        """
#        global_indeces_scurve=np.hstack((self.patch_data.patch_global_2D(2)[:,-1],self.patch_data.patch_global_2D(4)[:,-1]))
        self.boundary_pts_ess=np.hstack((bl_corner,br_corner,tr_corner,tl_corner))  
        boundary_pts_ess= self.boundary_pts_ess 
        u_hat_corners=np.ones(len(boundary_pts_ess))*40.*10**6

        """
        Example problem
        """
#        global_indeces_scurve=np.hstack((self.patch_data.patch_global_2D(2)[:,-1],self.patch_data.patch_global_2D(4)[:,-1]))
#        self.boundary_pts_ess=global_indeces_1[:,0]  
#        boundary_pts_ess= self.boundary_pts_ess 
#        u_hat_corners=np.ones(len(boundary_pts_ess))*40.*10**6
#        u_hat_corners=np.ones(len(boundary_pts_ess))*0
        
        
        
        """
        Applies flux to edge
        """
        for i in range(len(self.nuemann_bound)):
            if self.nuemann_bound[i,0]==-1:
                continue
            direction=self.nuemann_bound[i,0]; boundary=self.nuemann_bound[i,1]; patch=self.nuemann_bound[i,2]
            flux=self.flux[i]
            self.edge_integral(direction,boundary,patch,flux)
            
        """
        Applying bcs to non modified matrix
        """
        
        self.F+= -np.einsum('ij,j',self.K[:,boundary_pts_ess],u_hat_corners)
        self.K[:,boundary_pts_ess]=0
        self.K[boundary_pts_ess,:]=0
        self.K[boundary_pts_ess,boundary_pts_ess]=1.
        self.F[boundary_pts_ess]=u_hat_corners

#        
        """
        Applies flux to points
        """
#        boundary_pts_nat=np.hstack((bl_corner,br_corner,tr_corner,tl_corner))  
#        boundary_pts_nat=np.hstack((bl_corner,tr_corner))        
#
#        boundary_pts_nat=np.hstack((tm_point,bm_point,lm_point,rm_point))  
#        boundary_pts_nat=np.hstack((tm_point,bm_point,bl_corner,tl_corner,br_corner,tr_corner,lm_point,rm_point))   
#        boundary_pts_nat=np.hstack((global_indeces_3[:,-1],global_indeces_5[:,-1]))
#        flux_corners=np.array([1,-1])*self.source_strength        
##
##      
#        self.F[boundary_pts_nat]+=flux_corners
#      
      
        """
        Psuedocode for L.M. method
        """
#        ###Variable for size of G and lam matrices
#        number_global_bases=len(self.B)
#        
#        ###Indeces if edge control points
#        bottom_cp=self.patch_data.patch_global_2D(0)[0,:]
#        left_cp=self.patch_data.patch_global_2D(0)[:,0]
#        top_cp=self.patch_data.patch_global_2D(0)[-1,:]
#        
#        ###x vlaues for calulating pressure vals along top
#        top_x=self.B[np.array(top_cp),0]
#        
#        #u_hat entries in lam vector
#        bottom_lam=np.zeros(len(bottom_cp))
#        left_lam=np.zeros(len(left_cp))
#        top_lam=10**7*np.sin(np.pi*(top_x)/1600)
#        
#        #Initializing matrices
#        edge_cp=np.unique(np.hstack((bottom_cp,left_cp,top_cp)))
#        self.num_lambda=len(edge_cp)
#        self.G=np.zeros((number_global_bases,number_global_bases))
#        self.lam=np.zeros(number_global_bases)
#        
#        
#        ###Find matrix entries for bottom edge, then add to other edges. B/c elements are square and structured, this should work
#        xi_loc,wl=Ippre.compute_gauss_points_and_weights(np.max(self.order)+1,1) 
#        idx_grid = np.ix_(bottom_cp, bottom_cp)
#
#        for bb in range(self.number_elements[0,0]):
#            detJ,R,dR_dx=self.mappings(bb,xi_loc,np.ones(len(xi_loc))*-1,0)
#            xi,eta=self.parent_to_para(bb,xi_loc,np.ones(len(xi_loc))*-1,0)
#            detJ=np.ones(len(wl))*800/self.number_elements[0,0]/2
#            N=self.Bspline_basis(self.order[0,0],xi,self.patch_data.KV_xi_array[0],0,0)
#            R_N=np.einsum('ij,ik->jki',N,R[:,bottom_cp])
#           
#            N_u=np.einsum('ji,j->i',N,wl*detJ)
#            self.G[idx_grid]+=np.einsum('ijk,k->ij',R_N,wl*detJ)
#            self.lam[top_cp]+=N_u*top_lam
#            print(N_u)
##            self.lam[bottom_cp]+=
##        print(N_u)
#        common_vals=self.G[idx_grid]
#        
#        idx_grid2=np.ix_(left_cp, left_cp)
#        idx_grid3=np.ix_(top_cp, top_cp)
##        idx_grid3=np.ix_(top_cp, top_cp)
##        print(idx_grid2,idx_grid3)
#        self.G[idx_grid2]+=common_vals
#       
#        self.G[idx_grid3]+=common_vals
##        print(self.G)
##        print(self.lam)
#        self.G=-self.G[:,edge_cp]
#        self.lam=-self.lam[edge_cp]
        
#        self.lam[top_cp]+=
        #        bound_points,bound_wts=np.polynomial.legendre.leggauss(3)
#               
#        self.num_lambda=self.patch_data.num_bases[0,0]*4.-2. #Only works for current problem
#        lam_patch_0=np.linspace(0,(self.patch_data.number_elements[0,0]-1)+1,(self.patch_data.number_elements[0,0]-1)+2)
#        lam_patch_1=np.linspace(lam_patch_0[-1],lam_patch_0[-1]+(self.patch_data.number_elements[1,0]-1)+1,(self.patch_data.number_elements[1,0]-1)+2)
#        lam_patch_4=np.linspace(lam_patch_1[-1]+1,lam_patch_1[-1]+1+(self.patch_data.number_elements[4,0]-1)+1,(self.patch_data.number_elements[4,0]-1)+2)
#        lam_patch_5=np.linspace(lam_patch_4[-1],lam_patch_4[-1]+(self.patch_data.number_elements[5,0]-1)+1,(self.patch_data.number_elements[5,0]-1)+2)
#        
#        lam_patch_array=np.vstack((lam_patch_0,lam_patch_1,lam_patch_4,lam_patch_5))
##        return lam_patch_array
#        #Using convention consistent with paper
#        self.G=np.zeros((np.max(self.patch_data.Global_IEN)+1,self.num_lambda))
#        self.lam=np.zeros(self.num_lambda)
# 
#        bound_patches=np.array([0,1,4,5])
#        
#        #Iterate over patches
#        for pnum in bound_patches:
#            patch_num_elements_xi=self.patch_data.number_elements[pnum,0]
#            row_refs=self.patch_data.Global_IEN_length[pnum]
#            Global_IEN_patch=self.patch_data.Global_IEN[row_refs[0]:row_refs[1]+1]
#            local_IEN_patch=self.patch_data.local_IEN_array[pnum]
#            
#            if pnum==4:
#                lam_patch=lam_patch_array[2]
#            elif pnum==5:
#                lam_patch=lam_patch_array[3]
#            else:
#                lam_patch=lam_patch_array[pnum]
#               
#            #Selects localand global indeces of elements in patches occupying the boundaries
#            if pnum==0 or pnum==1:
#                Global_IEN_patch=Global_IEN_patch[0:patch_num_elements_xi]
#                local_IEN_patch= local_IEN_patch[0:patch_num_elements_xi]
#                uhat_bound=0
#            elif pnum==4 or pnum==5:
#                if len(local_IEN_patch)!=2:
#                    local_IEN_patch=local_IEN_patch[-patch_num_elements_xi:]
#                    Global_IEN_patch=Global_IEN_patch[-patch_num_elements_xi:]
##                return local_IEN_patch
#                uhat_bound=10.
#            
#            #Iterate over elements in patch
#            for enum in np.arange(patch_num_elements_xi): #iterate over elements in patch
#                lam_ele=(lam_patch[enum:enum+self.patch_data.order[pnum,0]+1]).astype(int)
#
#                element_global_indeces=Global_IEN_patch[enum,0:-2] #specific to current problem
#                element_local_indeces=local_IEN_patch[enum]
##                if pnum==4:
##                    return element_global_indeces
#                idx_grid = np.ix_(element_global_indeces,lam_ele) #right now, patch_lambdas defined over single element
#                
#                if pnum==4 or pnum==5:
#                    alt_enum=np.prod(self.number_elements[pnum])-self.number_elements[pnum,0]+enum
#                    
#                    
#                #Solves bases at integration points along boundary
#                for int_num in range(len(bound_wts)):
#                                        
#                    #Checks direction of boundary
#                    if pnum==0 or pnum==1:
#                        bound_eta_pt=-1
#                        xi,eta=self.parent_to_para(enum,bound_points[int_num],bound_eta_pt,pnum)
#                        detJ,R,dR_dx=self.mappings(enum,bound_points[int_num],bound_eta_pt,pnum) #can use regular mappings for detJ since patches in model problem are rectangles (detJ is constant)
#                        R=R[element_local_indeces]
#                        KV=self.patch_data.KV_xi_array[pnum]
#                        N=self.Bspline_basis(1,xi,KV,pnum,0)[enum:enum+self.patch_data.order[pnum,0]+1]
#                    else:
#                        bound_eta_pt=1
#                        xi,eta=self.parent_to_para(alt_enum,bound_points[int_num],bound_eta_pt,pnum)
#                        detJ,R,dR_dx=self.mappings(alt_enum,bound_points[int_num],bound_eta_pt,pnum) #can use regular mappings for detJ since patches in model problem are rectangles (detJ is constant)
#                        R=R[element_local_indeces]
#                        KV=self.patch_data.KV_xi_array[pnum]
#                        N=self.Bspline_basis(1,xi,KV,pnum,0)[enum:enum+self.patch_data.order[pnum,0]+1]
##                    if pnum==4 and int_num==1:
##                        return R,N                                           
##                    if pnum==4:
##                        return xi,eta,enum                                     
#                                  
#                    R_N_grid=np.einsum('i,j',R,N)
#
#                    self.G[idx_grid]+=R_N_grid*detJ*bound_wts[int_num]
#                    self.lam[lam_ele]+=uhat_bound*N*detJ*bound_wts[int_num]

#        return self.G,self.lam    
        
        
    def interface_cond(self,test=False):
        """
        Applies conditions to the interface. HARD CODED FOR CONSTANT SOURCE TERM ALONG INTERFACE.
        """
#        Integration points along each element side
        pts,wts=np.polynomial.legendre.leggauss(np.max(self.order)+5)   

        #Gathering global basis function with support on interface
        global_indeces=self.patch_data.patch_global_2D(2)[:,-1]
        
        local_indeces=self.patch_data.local_IEN(2,1)[:,-1]
#        Global_IEN_patch=self.patch_data.Global_IEN[self.patch_data.Global_IEN_length[2,0]:self.patch_data.Global_IEN_length[2,1],:]
#        IEN_patch=self.patch_data.local_IEN(2)
        source_strength_per_l=self.source_strength
       
        for ele_num in range(self.number_elements[2,1]):
            integrals=np.zeros(len(local_indeces))
            for int_pt in range(len(pts)):
                ele=(ele_num+1)* self.number_elements[2,0]-1       
                xi,eta=self.parent_to_para(ele,1,pts[int_pt],2)
                       
#                detJ,R,dR_dx=self.mappings(ele_num,1,pts[int_pt],2)
                detJ,R=self.mapping_interface(eta)
                
#                if int_pt==0 and ele_num==3:
#                    return detJ
                integrals+=R*wts[int_pt]
#                if ele_num==3:
#                    print(integrals,R)
                self.F[global_indeces]+=source_strength_per_l*R*detJ*wts[int_pt]
                
        if test==True:
            return integrals,detJ,R
#        return self.F[global_indeces]
#        control_point_load=np.array([22,20,18,16,14,12,10,8,4])-1 #should be basis function load
#        fhat=np.ones(9)*10
#        self.F[control_point_load]+=fhat  
        
    def create_solution(self):
        """
        Finds approximation at control points .
        
        Currently formatted for bc's at corners
        """
        #Runs functions to assemble matrix and vector with boundary conditions
        self.assemble()
        self.boundary_conditions()
#        self.interface_cond() #No interface in verification
        
        
        """
        """
#        Assemble b.c. modified global stiffness and force matrices
#        filler=np.zeros((self.num_lambda,self.num_lambda))
#        self.K=np.hstack((self.K,self.G))
#        self.K=np.vstack((self.K,np.hstack((np.transpose(self.G),filler))))
#        self.F=np.hstack((self.F,self.lam))
####
        K_s=csr_matrix(self.K.tolist())
        self.solution=spsolve(K_s,self.F)/10**6
#        self.solution=spsolve(K_s,self.F)
#        self.solution=self.solution[0:-self.num_lambda]
             
    def approx_function(self,xi,eta,patch_num):
        """
        Finds solution at given xi_til,eta_til, and patch number
        """
        
        approx=self.solution        
        refs=self.patch_data.patch_global_2D(patch_num)
        refs=refs.flatten()
        approx=approx[refs]
        
        xi_vec=np.array([xi,0]); eta_vec=np.array([eta,0])
        patch_bases=self.NURBS_bases(xi_vec,eta_vec,patch_num)
     
        return np.dot(approx,patch_bases[0,:]) #TEMPORARY FIX


        
    def plotter(self,num_pts,mesh=True,dof=True):       
        """
        Plots solution by interpolating between nodes
        """  
        xi_til_pts=np.linspace(-1,1,num_pts)
        eta_til_pts=np.linspace(-1,1,num_pts) 
        Xi_grid,Eta_grid=np.meshgrid(xi_til_pts,eta_til_pts)
        Xi_grid=Xi_grid.flatten()
        Eta_grid=Eta_grid.flatten()
        

        max_press=max(self.approx_function(0,0,0),self.approx_function(1,0,1),self.approx_function(0,1,4),self.approx_function(1,1,5),self.approx_function(1,1,2),self.approx_function(0,1,5),self.approx_function(0,1,2))
        min_press=min(self.approx_function(1,0,0),self.approx_function(0,0,0),self.approx_function(1,0,1),self.approx_function(0,1,4),self.approx_function(1,1,5),self.approx_function(1,1,2),self.approx_function(1,.5,4),self.approx_function(1,.25,4),self.approx_function(1,1,7))

      
#        max_press=max(self.approx_function(0,0,0),self.approx_function(1,0,0),self.approx_function(0,1,0),self.approx_function(1,1,0))
#        min_press=min(self.approx_function(0,0,0),self.approx_function(1,0,0),self.approx_function(0,1,0),self.approx_function(1,1,0))
        
        approx=self.solution[0:len(self.B)] #ignores lambdas in solution]
        solution_grid_total=np.zeros((num_pts,num_pts,np.sum(self.number_elements[:,0]*self.number_elements[:,1])))
        x_pts_grid_total=np.zeros((num_pts,num_pts,np.sum(self.number_elements[:,0]*self.number_elements[:,1])))
        y_pts_grid_total=np.zeros((num_pts,num_pts,np.sum(self.number_elements[:,0]*self.number_elements[:,1])))
        
        sol_sum=np.zeros(num_pts**2)
        count=-1
        if np.array_equal(self.patch_data.order[1,:],np.ones(2)*-1)==0:
            num_patches=len(self.number_elements)
        else:
            num_patches=1

        #Iterates over patches
        for aa in range(num_patches):
#            if self.order[aa,0]==-1:
#                break
            IEN_patch=self.patch_data.local_IEN_array[aa]
            
            if num_patches==1:
                Global_IEN_patch=self.patch_data.Global_IEN
            else:
                Global_IEN_patch=self.patch_data.Global_IEN[self.patch_data.Global_IEN_length[aa,0]:self.patch_data.Global_IEN_length[aa,1]+1,0:np.prod(self.order[aa]+1)]

            control_point_ref=self.patch_data.patch_global_2D(aa).flatten()
            B_patch=self.B[control_point_ref]

            #Iterate over elements
            for bb in range(np.prod(self.number_elements[aa])):
                Global_IEN_element=Global_IEN_patch[bb,:]        
                
                #Maps xi_til, eta_til to xi and eta
                xi,eta=self.parent_to_para(bb,Xi_grid,Eta_grid,aa) #find xi and eta for given xi_til and eta_til
                
                #Evaluates approximation at meshgrid point 
                R=self.NURBS_bases(xi,eta,aa)
                
                count+=1
                    
#                    print(R)
#                    print(R[:,IEN_patch[bb,:]])
#                    solution_grid_total[:,:,count]=np.dot(approx[Global_IEN_element],R[lvl,IEN_patch[bb,:]])
#                    a=R[:,IEN_patch[bb,:]]*approx[Global_IEN_element]
                sol_sum=np.sum(R[:,IEN_patch[bb,:]]*approx[Global_IEN_element],axis=1)
#                    print(R[:,IEN_patch[bb,:]]*approx[Global_IEN_element])
                sol_sum=sol_sum.reshape(num_pts,num_pts)
                solution_grid_total[:,:,count]=sol_sum
                #Maps xi and eta to x and y
                xfun=np.sum(R*B_patch[:,0],axis=1)
                yfun=np.sum(R*B_patch[:,1],axis=1)
                xfun=xfun.reshape(num_pts,num_pts); yfun=yfun.reshape(num_pts,num_pts)
                x_pts_grid_total[:,:,count]=xfun
                y_pts_grid_total[:,:,count]=yfun
                    
        
        #Plots data patchwise
        ticks=np.linspace(min_press,max_press,5)
#        fig,ax=plt.subplots()
        fig=plt.figure()
        ax1=fig.add_subplot(111)
#        return x_pts_grid_total
#        if num_patches>1:
        for i in range(np.shape(x_pts_grid_total)[2]):
            cs=ax1.contourf(x_pts_grid_total[:,:,i],y_pts_grid_total[:,:,i],solution_grid_total\
            [:,:,i],levels=np.linspace(min_press,max_press,30),cmap="coolwarm")   
                 
        axes().set_aspect('equal')
        props = dict(boxstyle='square', facecolor='white', alpha=1.)
#        plt.text(.5,.9,'Degrees of freedom= %s' %len(self.F), ha='center', va='top',bbox=props)
#        plt.xticks(np.linspace(-.5,1,4))
#        plt.yticks(np.linspace(0,1.5,3))
        cbar=fig.colorbar(cs,ax=ax1, orientation='vertical',ticks=ticks)
        props = dict(boxstyle='square', facecolor='white', alpha=1.)
        
        if dof==True:
            plt.text(0.5, 0.9,'Degrees of freedom= %s' %len(self.F), ha='center', va='center', transform=ax1.transAxes,bbox=props,fontsize=16)
        plt.xlabel('x [m]',fontsize=18)
        plt.ylabel('y [m]',fontsize=18)
        plt.xticks(np.linspace(-400,400,5))
        plt.yticks(np.linspace(-400,400,5))
        plt.gcf().subplots_adjust(bottom=0.15)   
#        plt.xticks(np.linspace(-.5,1,4))
#        plt.yticks(np.linspace(0,1.5,3))
        
        plt.tick_params(labelsize=18)
        cbar.ax.tick_params(labelsize=18)
        cbar.set_label('Pressure [MPa]', labelpad=10, size=18)
        
        if mesh==False:
#            return plt,solution_grid_total,x_pts_grid_total,y_pts_grid_total,xi,eta
            return plt
        """
        Overlay the element boundaries
        """
        pts_vec=np.linspace(0,1,100)
        for patch in range(len(self.order)):
            if self.order[patch,0]==-1:
                continue
            xi_vector=self.patch_data.KV_xi_array[patch]
            eta_vector=self.patch_data.KV_eta_array[patch]
            control_points=self.B[self.patch_data.patch_global_2D(patch).flatten()]
            
#            Xi_grid,xi_pts_grid=np.meshgrid(xi_vector,pts_vec)
#            Xi_grid=Xi_grid.flatten(); xi_pts_grid=xi_pts_grid.flatten()
#            
#            Eta_grid,eta_pts_grid=np.meshgrid(eta_vector,pts_vec)
#            Eta_grid=Eta_grid.flatten(); eta_pts_grid=eta_pts_grid.flatten()
#            
#            R_xi=self.NURBS_bases(Xi_grid,xi_pts_vec,patch)
#            x=np.sum(control_points[:,0]*R_xi,axis=1)
#            y=np.sum(control_points[:,1]*R_xi,axis=1)
#            
#            R_eta=self.NURBS_bases(eta_pts_grid,Eta_grid,patch)
#            y=np.sum(control_points[:,1]*R_eta,axis=1)
            
            
            for xi in range(len(xi_vector)):
                x=np.sum(control_points[:,0]*self.NURBS_bases(np.ones(len(pts_vec))*xi_vector[xi],pts_vec,patch),axis=1)
                y=np.sum(control_points[:,1]*self.NURBS_bases(np.ones(len(pts_vec))*xi_vector[xi],pts_vec,patch),axis=1)
                if xi_vector[xi]==0 or xi_vector[xi]==1:
                    plt.plot(x,y,'k',alpha=1,linewidth=2)
                else:
                    plt.plot(x,y,'k',alpha=.25)
#               
            for eta in range(len(eta_vector)):
                x=np.sum(control_points[:,0]*self.NURBS_bases(pts_vec,np.ones(len(pts_vec))*eta_vector[eta],patch),axis=1)
                y=np.sum(control_points[:,1]*self.NURBS_bases(pts_vec,np.ones(len(pts_vec))*eta_vector[eta],patch),axis=1)
                if eta_vector[eta]==0 or eta_vector[eta]==1:
                    plt.plot(x,y,'k',alpha=1,linewidth=2)
                else:
                    plt.plot(x,y,'k',alpha=.25)
#        plt.scatter(self.B[:,0],self.B[:,1])
 
        return plt
        
    def plotter_mesh(self):
        
        fig,ax=plt.subplots()
        cmap = matplotlib.cm.get_cmap('coolwarm')
        pts_vec=np.linspace(0,1,100)
        for patch in range(len(self.order)):
            xi_vector=self.patch_data.KV_xi_array[patch]
            eta_vector=self.patch_data.KV_eta_array[patch]
            control_points=self.B[self.patch_data.patch_global_2D(patch).flatten()]

            for xi in range(len(xi_vector)):
                x=np.sum(control_points[:,0]*self.NURBS_bases(np.ones(len(pts_vec))*xi_vector[xi],pts_vec,patch),axis=1)
                y=np.sum(control_points[:,1]*self.NURBS_bases(np.ones(len(pts_vec))*xi_vector[xi],pts_vec,patch),axis=1)
                if xi_vector[xi]==0 or xi_vector[xi]==1:
                    plt.plot(x,y,'k',alpha=1,linewidth=2)
                else:
                    plt.plot(x,y,'k',alpha=.25)
          
    
            for eta in range(len(eta_vector)):
                x=np.sum(control_points[:,0]*self.NURBS_bases(pts_vec,np.ones(len(pts_vec))*eta_vector[eta],patch),axis=1)
                y=np.sum(control_points[:,1]*self.NURBS_bases(pts_vec,np.ones(len(pts_vec))*eta_vector[eta],patch),axis=1)
                if eta_vector[eta]==0 or eta_vector[eta]==1:
                    plt.plot(x,y,'k',alpha=1,linewidth=2)
                else:
                    plt.plot(x,y,'k',alpha=.25)
        ax.set_aspect('equal')
        plt.scatter(self.B[:,0],self.B[:,1],color='b',marker='s',s=30,linewidth=3)
        
        """
        Data for example problem
        """
#        plt.xticks(np.linspace(-.5,1,4))
#        plt.yticks(np.linspace(0,1.5,3))
#        plt.tick_params(labelsize=16)
#        plt.xlabel('x [m]',fontsize=16)
#        plt.ylabel('y [m]',fontsize=16)
        
        
        return plt


