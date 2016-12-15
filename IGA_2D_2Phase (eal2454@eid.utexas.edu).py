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
#pgf_with_custom_preamble = {
#    "font.family": "serif",   # use serif/main font for text elements
#    "text.usetex": True,    # use inline math for ticks
#    "pgf.rcfonts": False,   # don't setup fonts from rc parameters
#}
#matplotlib.rcParams.update(pgf_with_custom_preamble)
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import IGA_refine  
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix
from scipy.optimize import newton_krylov, fsolve, anderson
from scipy.optimize import broyden1
import matplotlib.animation as animation
import time
from PIL import Image
import os
from Buckley_Leverett import Buckley_Leverett
#import cffi
#ffi = cffi.FFI()
#
#
#ffi.cdef("""    
#            void evaluate_u2_IGA(const int p, int num_of_eval_pts, const int length, const int num_knots, 
#                       const double* knot_vector, const double* solution, 
#                       double* x, double* u2);
#                       
#            void evaluate_u1_IGA(const int p, const int num_of_eval_pts, const int length, const int num_knots, 
#                       const double* knot_vector, const double* solution, 
#                       double* x, double* u1);
#                       
#            void evaluate_lambda_IGA(const int p, const int num_of_eval_pts, const int length, const int num_knots, 
#                       const double* knot_vector, const double* solution, 
#                       double* x, double* lambda);
#                   
#            void evaluate_constraint_IGA(const int p, const int num_of_eval_pts, const double length, const int num_knots, 
#                       const double* knot_vector, const double* solution, 
#                       double* x, double* constraint);
#                   
#            void bspline(const int p, const int num_of_eval_pts, const double length, const int num_knots, 
#                   const double* knot_vector, double* x, double* spline);
#            
#            void computeIGAResidual(const int p, const bool constraint_flag, const double youngs_mod, const double area, const double moment_of_inertia, 
#                        const double P1_location, const double P2_location, const double P1, const double P2,
#                        const int int_rule, const double* int_points, const double* int_wts, 
#                        const int num_knots, const double* knot_vector,
#                        const double* unknowns, double* residual);
#                        
#            void computeIGAProjectionResidual(const int p, const double youngs_mod, const double area, const double moment_of_inertia, 
#                        const double P1_location, const double P2_location, const double P1, const double P2,
#                        const int int_rule, const double* int_points, const double* int_wts, 
#                        const int num_knots, const double* knot_vector,
#                        const double* unknowns, double* residual);
#            """)
            
class IGA_2D_2Phase():
    
    """
    Solves 2D boundary value problem with line sources/sinks using IGA.
    """
    
    def __init__(self,mesh_data,control_net,permeability,reservoir_properties,rel_perm_params,fluid_properties,times,initial_pressure,initial_saturation, \
    nuemann_boundaries=None,nuemann_values=None,dirichlet_boundaries=None,dirichlet_values=None,source_locations_flux=None, \
    source_values_flux=None,source_locations_pressure=None,source_values_pressure=None,boundary_saturations_locations=None,boundary_saturations_values=None,dimension=2):
            
        """
        INPUTS:
        
        order:               matrix of bases orders in xi and eta for each patch
        number_elements:     number of bases in xi and eta for eac patch
        mulitplicity:        uniform multiplicity of interna knots in xi and eta for each patch
        patch_connect_info:  Array describing along which common boundaries patches are found. 
                             See furthur description in IGA_pre_processing.py
        B:                   Control net
        K:                   perm tensor
        mu:                  viscosities
        nuemann_bound:       Boundary data for iniitial nuemann condition
        flux:                Corresponding flux values for nuemann boundaries
        init_cond:           Initial water pressure and saturation at control points. (will only do uniform properties for now)   
        dirichlet_bound:     Boundary data for initial dirichlet conditons
        ess_vals:            Pressure and saturation values at dirichlet nodes
        times:               Time values for which pressures and saturations are stored
        """
        
        """
        Does not include forcing terms. Two phase flow of oil and water
        """
        
        """
        Initialization of terms. Will have to include natural condition inpputs in future versions
        """
        
        ###Mesh data 
        self.patch_data=mesh_data
        self.order=mesh_data.order; self.number_elements=mesh_data.number_elements; self.multiplicity=mesh_data.mp; self.patch_connect_info=mesh_data.patch_connect_info
        self.B=control_net
    
        ###Reservoir properties
        self.k=permeability; self.phi=reservoir_properties[0]; self.lam=reservoir_properties[1]; self.S_wirr=reservoir_properties[2]; self.S_m=reservoir_properties[3]
        
        ###Rel perm params
        self.k_wr=rel_perm_params[0]; self.k_nwr=rel_perm_params[1]; self.lam_1=rel_perm_params[2]; self.lam_2=rel_perm_params[3]       
        
        ###Fluid Properties
        self.rel_perm_params=rel_perm_params; self.mu_w=fluid_properties[0]; self.mu_o=fluid_properties[1]; self.B_w=fluid_properties[2]; self.B_o=fluid_properties[3]
        
        ###Times of sampling
        self.times=times
        
        ###Boundary conditions
        self.initial_pressure=initial_pressure
        self.initial_saturation=initial_saturation
        self.nuemann_boundaries=nuemann_boundaries
        self.nuemann_values=nuemann_values  
        self.dirichlet_boundaries=dirichlet_boundaries
        self.dirichlet_values=dirichlet_values
        self.source_locations_flux=source_locations_flux
        self.source_values_flux=source_values_flux
        self.source_locations_pressure=source_locations_pressure
        self.source_values_pressure=source_values_pressure
        self.boundary_saturations_locations=boundary_saturations_locations
        self.boundary_saturations_values=boundary_saturations_values
        
        self.dimension=dimension
        """
        Pre compute solution
        """
        self.SG=0
#        if self.SG==0:
#            self.eps_w=0
#        else:
#        self.eps_w=.0001
        self.create_solution(mode='IMPES')
        
    def parent_to_para(self,patch_num,e,xi_tilde,eta_tilde=None):
        """
        Maps from parent element to parametric space
        
        INPUTS:
        e:           element number (Python indeces)
        xi_tilde:    xi_tilde coordinate in parent space
        eta_tilde:   eta_tilde coordinate in parent space
        patch_num:   patch number (Python indeces)
        """
        """
        1D case
        """
        if self.dimension==1:
            IEN_patch=self.patch_data.local_IEN_array[patch_num]
            INC_patch=self.patch_data.INC_array[patch_num]
            KV_xi_patch=self.patch_data.KV_xi_array[patch_num]
            
            if len(IEN_patch.shape)!=1:
                ni=INC_patch[IEN_patch[e,0]] #CHAnGED DEF OF e TO BE CONSISTENT WITH PYTHON. IF IEN_PATCH IS !D ARRAY CANT USE 2 INDECES
            else:
                ni=INC_patch[IEN_patch[0]] #CHAnGED DEF OF e TO BE CONSISTENT WITH PYTHON. IF IEN_PATCH IS !D ARRAY CANT USE 2 INDECES
  
            xi=((KV_xi_patch[ni+1]-KV_xi_patch[ni])*xi_tilde+(KV_xi_patch[ni+1]+KV_xi_patch[ni]))/2.
            return xi
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
        
    def alphas(self,degree,basis_numbers,KV_patch,deriv_order):
        """
        Computes alphas needed for kth order derivative (see pg 28 of IGA book)
        """
        i=basis_numbers.astype('int')
        alphas=np.zeros((deriv_order+1,len(i)))
        denominator=np.zeros((deriv_order+1,len(i)))
        if deriv_order==0:
            alphas=np.ones((1,len(i)))
#            alphas=alphas[0,:]
#            print (alphas)
        else:
#            numerator=np.zeros((1,len(i)))
            for j in np.linspace(0,deriv_order,deriv_order+1).astype('int'):
#                numerator=np.zeros((j+1,len(i)))
#                print(numerator)
                if j==0:
                    numerator=self.alphas(degree,basis_numbers,KV_patch,deriv_order-1)[j,:]
                    denominator[j,:]=(KV_patch[i+degree-deriv_order+1]-KV_patch[i])
                elif (j!=0 and j!=deriv_order):
                    numerator=np.vstack((numerator,(self.alphas(degree,basis_numbers,KV_patch,deriv_order-1)[j,:]-self.alphas(degree,basis_numbers,KV_patch,deriv_order-1)[j-1,:])))
                    denominator[j,:]=(KV_patch[i+degree+j-deriv_order+1]-KV_patch[i+j])
                elif j==deriv_order:
                    numerator=np.vstack((numerator, -self.alphas(degree,basis_numbers,KV_patch,deriv_order-1)[j-1,:]))
                    denominator[j,:]=(KV_patch[i+degree+j-deriv_order+1]-KV_patch[i+j])

            
            with np.errstate(divide='ignore', invalid='ignore'):
                denominator = np.where(denominator != 0.0, 
                                      (1/denominator), 0.0)
#            print(numerator,denominator)
            alphas=np.einsum('ij,ij->ij',numerator,denominator)    
#        print (alphas)
    
        return alphas
                
    def Bspline_basis(self,degree,xi,KV_patch,patch_num,xi_dir=0,deriv_order=0):
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
#        print(xi,deriv_order)
        if degree == 0 :
            return self.Bspline_basis0(xi,KV_patch)
        else:
            basis_p_minus_1= self.Bspline_basis(degree-1,xi,KV_patch,patch_num,xi_dir)
        
        """
        Checks if direction is in xi or eta. Chooses p (xi degree) or q (eta degree) accordingly
        """
        if xi_dir==0:
            max_degree=self.order
        else:
            max_degree=self.order   
        
        """
        Assembles terms to be used in Bspline bases
        """

        first_term_numerator = xi[:, np.newaxis] - KV_patch[:-degree] 
        first_term_denominator = KV_patch[degree:] - KV_patch[:-degree]
        
        second_term_numerator = KV_patch[(degree + 1):] - xi[:, np.newaxis]
        second_term_denominator = (KV_patch[(degree + 1):] - 
                                  KV_patch[1:-degree])
        
        """
        Change numerator in last recursion if derivatives are desired
        """
#        alpha=np.array([1])
#        for j in range(deriv_order):
#            alpha
        
#        if compute_derivatives and degree == max_degree:
#            
#            first_term_numerator = np.ones((len(xi), 
#                                            len(first_term_denominator))) * degree
#            second_term_numerator = np.ones((len(xi), 
#                                             len(second_term_denominator))) * -degree

#            
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
        if deriv_order==0:
            return a
        else:
            kth_deriv=np.zeros((len(xi),len(self.B)))
            if degree-deriv_order>0:
                C=math.factorial(degree)/math.factorial(degree-deriv_order)  
            else:
                C=math.factorial(degree)
#            print(degree-deriv_order)
            bases=self.Bspline_basis(degree-deriv_order,xi,KV_patch,patch_num,xi_dir,0)
            
            alphas=self.alphas(degree,np.arange(a.shape[1]),KV_patch,deriv_order)
#            print(bases[:,0:(0+deriv_order+1)])
#            print(kth_deriv.shape)
            for col in range(kth_deriv.shape[1]):
                kth_deriv[:,col]+=C*np.einsum('i,ji->j',alphas[:,col],bases[:,col:(col+deriv_order+1)])       
                                        
            return kth_deriv

     
    def NURBS_bases(self,patch_num,xi,deriv_order=0,deriv='all',test=False):
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
        1D case
        """
        if self.dimension==1:
            control_point_ref=self.patch_data.patch_global_2D(patch_num).flatten()
            weights=self.B[control_point_ref,-1]
            KV_xi_patch=self.patch_data.KV_xi_array[patch_num]
            p=self.order
            N=self.Bspline_basis(p,xi,KV_xi_patch,patch_num,xi_dir=0,deriv_order=0)
            R=np.zeros((len(xi),len(N[0,:])))
          
            for row in range(len(xi)):
                R[row,:]=N[row,:]*weights #ravel vs. flatten?      
                
            sum_tot=np.sum(R,axis=1)
           
            for row in range(len(xi)):
                R[row,:]=R[row,:]/sum_tot[row]    
            
            if deriv_order==0:
                return R.T
            if deriv_order>self.order and deriv=='all':
#                print(np.shape(R))
                return R.T,np.zeros(np.shape(R)).T
            else:
                binom=lambda j,k: math.factorial(k)/(math.factorial(j)*(math.factorial(k-j)))
#                print(p,xi,deriv_order)
                dN_dxi=self.Bspline_basis(p,xi,KV_xi_patch,patch_num,xi_dir=0,deriv_order=deriv_order)
#                print(dN_dxi)
                A_k=np.einsum('ij,j->ij',dN_dxi,weights)
                
                dR_dxi=np.zeros((len(xi),len(N[0,:]))) 
                
                W=np.einsum('ij,j->i',N,weights)
                
                numerator_term=np.zeros((len(xi),len(N[0,:]))) 
                for j in np.arange(1,deriv_order+1):
                    bi=binom(j,deriv_order)

                    R_c=self.NURBS_bases(patch_num,xi,deriv_order=deriv_order-j,deriv='dR_dxi')
                    R_c=R_c.T

                    W_j=self.Bspline_basis(p,xi,KV_xi_patch,patch_num,xi_dir=0,deriv_order=j)

                  
                    W_j=np.einsum('ij,j->i',W_j,weights)

                    numerator_term+=np.einsum('ij,i->ij',bi*R_c,W_j)

                dR_dxi=np.einsum('ij,i->ij',(A_k-numerator_term),1./W)
                
                    
                
                
                if deriv=='dR_dxi':
                    return  dR_dxi.T
                elif deriv=='all':
                    return R.T,dR_dxi.T
      
                
    def mappings(self,patch_num,e,xi_tilde,deriv_order=1,test=False):
        """
        Finds Jacobian for mapping from parent to physical space. 
        
        INPUTS:
        
        e:          element number (Python indeces)
        xi_tilde:   xi_tilde coordinate in parent space
        eta_tilde:   eta_tilde coordinate in parent space
        patch_num:  patch_number (Python indeces)
        """
        """
        1D case
        """
        if self.dimension==1:
            xi=self.parent_to_para(patch_num,e,xi_tilde,None)
#            print(xi)
            IEN_patch=self.patch_data.local_IEN_array[patch_num]
            INC_patch=self.patch_data.INC_array[patch_num]
            KV_xi_patch=self.patch_data.KV_xi_array[patch_num]
            control_point_ref=self.patch_data.patch_global_2D(patch_num).flatten()
            B_patch=self.B[control_point_ref,:]
            
            if len(IEN_patch.shape)!=1:
                ni=INC_patch[IEN_patch[e,0]] #bottom left corner of element
            else:
                ni=INC_patch[IEN_patch[0]] #bottom left corner of element
            
            if deriv_order==1:
                R,dR_dxi=self.NURBS_bases(patch_num,xi,1,'all')
                """
                Mapping from parameter to physical space
                """
                dx_dxi=np.zeros(len(xi)) #for 2D
                dxi_dx=np.zeros(len(xi)) #for 2D
                dR_dx=np.zeros((len(R[:,0]),len(xi)))
#                print(dR_dxi)
                for lvl in range(len(xi)):
                    dx_dxi[lvl]=np.dot(B_patch[:,0],dR_dxi[:,lvl])
                
                """
                Compute inverse of gradient
                """
                dxi_dx=1/dx_dxi
#                print(dxi_dx)
                """
                Compute derivatives of basis functions with respect to physical coordinates     
                """
                for lvl in range(len(xi)):
                    dR_dx[:,lvl]=dR_dxi[:,lvl]*dxi_dx[lvl]
#                print(dR_dx)
            if deriv_order==2:
                R,dR_dxi=self.NURBS_bases(patch_num,xi,1,'all')
                R,dR_dxi2=self.NURBS_bases(patch_num,xi,2,'all')
#                print(R,dR_dxi2)
                """
                Mapping from parameter to physical space
                """
                dx_dxi=np.zeros(len(xi)) #for 2D
                dxi_dx=np.zeros(len(xi)) #for 2D
                dx_dxi2=np.zeros(len(xi)) #for 2D
                dxi_dx2=np.zeros(len(xi)) #for 2D
                dR_dx2=np.zeros((len(R[:,0]),len(xi)))
                
                for lvl in range(len(xi)):
#                    print (dR_dxi2)
                    dx_dxi[lvl]=np.dot(B_patch[:,0],dR_dxi[:,lvl])
                    dx_dxi2[lvl]=np.dot(B_patch[:,0],dR_dxi2[:,lvl])
#                print(dx_dxi2)

                """
                Compute inverse of gradient
                """
                dxi_dx=1/dx_dxi
                dxi_dx2=1/dx_dxi2
                
                """
                Compute derivatives of basis functions with respect to physical coordinates     
                """
                for lvl in range(len(xi)):
                    dR_dx2[:,lvl]=dR_dxi2[:,lvl]*dxi_dx[lvl]
#                    +dR_dxi[:,lvl]*dxi_dx2[lvl]
                    
                return dR_dx2
            """
            Gradient of mapping from parent element to parameter space
            """
            dxi_dtildexi=0 #2x2 for 2D
            J=np.zeros([len(xi)]) #Jacobian matrix
            detJ=0
            
            if len(IEN_patch.shape)!=1:
                ni=INC_patch[IEN_patch[e,0]] #bottom left corner of element
            else:
                ni=INC_patch[IEN_patch[0]] #bottom left corner of element
                
            dxi_dtildexi=(KV_xi_patch[ni+1]-KV_xi_patch[ni])/2. #check mapping
            
            """
            Gradient of mapping from parent to physical space  
            """
            J=dx_dxi*dxi_dtildexi

            detJ=J
            
            #Return same output as NURBS_bases to save computation time in assembly
            if test==False:
                return detJ,R,dR_dx
            else:
                return dx_dxi,dR_dxi
        
    
    def relative_permeability(self,Sw,deriv=0):
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
        Sw_norm=(Sw-Sw_irr)/(S_m-Sw_irr)
        krw=self.k_wr*(Sw_norm)**self.lam_1
        krw1=(self.k_wr*self.lam_1)*Sw_norm**(self.lam_1-1)*1/(S_m-Sw_irr)
        kro=self.k_nwr*(1-Sw_norm)**self.lam_2
        kro1=(self.k_nwr*self.lam_2)*(1-Sw_norm)**(self.lam_2-1)*(-1)/(S_m-Sw_irr)
        
        if deriv==0:
            return krw, kro
        else:
            return krw1,kro1
            
    def capillary_pressure(self,Sw,deriv=0):
        """
        Finds capillary pressure as a function of water saturation
        """
        P_e=1.; 
        lam=100.;
        Sw_norm=(Sw-self.S_wirr)/(self.S_m-self.S_wirr)
        P_c=P_e*((Sw_norm)**(-1/lam)-1)    #P_e=entry pressure
        dPc_dSw=P_e*(-1/lam)*(Sw_norm)**(-1/lam-1)*1/(self.S_m-self.S_wirr)
        dPc_dSw2=P_e*(-1/lam)*1/(self.S_m-self.S_wirr)**2*(-1/lam-1)*(Sw_norm)**(-1/lam-2)
        with np.errstate(divide='ignore', invalid='ignore'):
            dPc_dSw = np.where(Sw_norm<=0,0,dPc_dSw)

        if deriv==0:
            return P_c
        elif deriv==1:
            return dPc_dSw
        elif deriv==2:
            return dPc_dSw2
    
    def F_w(self,Sw,deriv=0):
        """
        Finds fractional flows
        """
        krw,kro=self.relative_permeability(Sw);  dkrw_dSw,dkro_dSw=self.relative_permeability(Sw,1);
#        q=self.source_values_flux
#        print(kro)
        
        fw=1./(1.+kro*self.mu_w/(krw*self.mu_o))
        with np.errstate(divide='ignore', invalid='ignore'):
            fw = np.where(fw<0,0,fw)
        if deriv==0:
            return fw
        else:
            dfw_dSw=-(1+(kro*self.mu_w)/(krw*self.mu_o))**-2*self.mu_w/self.mu_o*(dkro_dSw/krw-kro*krw**-2*dkrw_dSw)
            return dfw_dSw
    def tau(self,patch,ele_num,xi_pts,element_indeces,saturation,pressure,Q=False):
        """
        Finds stabilization parameters for given integration points
        """
        ###Basis functions evaluated at integration points
        detJ,R,dR_dx=self.mappings(patch,ele_num,xi_pts)
        dR_dx2=self.mappings(patch,ele_num,xi_pts,2)
        
        ###Only taking bais functions with support on current element 
        dR_dx=dR_dx[element_indeces,:]; dR_dx2=dR_dx2[element_indeces,:];

        ###Finding nodal values of saturation and pressure on element
        Sw=saturation[element_indeces];p=pressure[element_indeces];
        
        ###Finding sats and pressures at integration points
        Sw_pts=np.einsum('ij,i->j',dR_dx,Sw);
        
        ###Finding relative permeabilties and their derivatives wrt Sw
        krw,kro=self.relative_permeability(Sw_pts);  dkrw_dSw,dkro_dSw=self.relative_permeability(Sw_pts,1);

        ###Cappilary pressures and ththeir derivatives wrt Sw        
        pc=self.capillary_pressure(Sw_pts);  dpc_dSw=self.capillary_pressure(Sw_pts,1);  dpc_dSw2=self.capillary_pressure(Sw_pts,2)
       
        ###Finding mobility and it's derivatve
        lamw=self.k*krw/self.mu_w/self.phi; dlamw_dSw=self.k*dkrw_dSw/self.mu_w/self.phi; 

        ###Derivative of pressure with respect to x
        dp_dx=np.einsum('ij,i->j',dR_dx,p); dp_dx2=np.einsum('ij,i->j',dR_dx2,p);
        
        ###Derivative of x wrt Sw
        dSw_dx=np.einsum('ij,i->j',dR_dx,Sw);dx_dSw=1/dSw_dx; dSw_dx2=np.einsum('ij,i->j',dR_dx2,Sw)
        
        ###Finding water fraction and its approximate derivative (taking derivative of Fw rather than fw)
        fw=self.F_w(Sw_pts)
        dfw_dSw=self.F_w(Sw_pts,1)
       
        ###Finding stabilization parameters
        A1=dlamw_dSw*dp_dx+lamw*dp_dx2*dx_dSw
        A2=self.k/self.phi*self.eps_w*dfw_dSw*dSw_dx
        A=A1-A2

        D=-self.eps_w*(1-fw)*self.k/self.phi
        h=self.B[-1,0]/self.number_elements

        v=np.abs(A); 
        eps=np.abs(D);
        
        alpha=1./2.*v*h/eps
        zeta=1/np.tanh(alpha)-1/alpha
        
        tau=1./2.*h*zeta/v
        tau=np.nan_to_num(tau)

        
        ###Finding linear ad-diff operator matrices
        L_1=np.einsum('ij,j->ij',dR_dx,A)
        L_2=np.einsum('ij,j->ij',dR_dx,A2)+np.einsum('ij,j->ij',dR_dx2,D)
        L=-L_1-L_2
        L=np.nan_to_num(L)
#        print(A2)
#        print(L)
        
#        jhb
#        print(tau)

        if Q==False:
            return tau,L
        else:
            return tau,D*dR_dx
    def assemble(self,saturation,pressure,delta_t,capillary_pressure=False,test=False,mode='implicit'):
        """
        Builds global  matrices. Follows convention in Kukreti 1989. 
        
        Y= pressures and saturations from current time step
        Yplus= pressures and 
        """
        nelem=self.number_elements
    
        number_global_bases=len(self.B) #number global basis functions
        
        """
        1D case
        """
        if self.dimension==1:
            
            number_patches=1
            """
            Integration points. May need to add additional points for log funcitons
            """
            xi_loc,wl=Ippre.compute_gauss_points_and_weights(np.max(self.order)+1,1) 

            """
            Initializing matriices. Matrices are components of the overall equations forming the residuals. 
            """
            
            K1=np.zeros((number_global_bases,number_global_bases))    
            K2=np.zeros((number_global_bases,number_global_bases))
            K3=np.zeros((number_global_bases,number_global_bases))
            empt_mat=np.zeros((number_global_bases,number_global_bases))
            
            A=np.zeros((number_global_bases,number_global_bases))
            B=np.zeros((number_global_bases,number_global_bases))
            C=np.zeros((number_global_bases,number_global_bases))
            D=np.zeros((number_global_bases,number_global_bases))

            A_VMS=np.zeros((number_global_bases,number_global_bases))
            B_VMS=np.zeros((number_global_bases,number_global_bases))
            B_VMS_tau=np.zeros((number_global_bases,number_global_bases))
            C_VMS=np.zeros((number_global_bases,number_global_bases))
            D_VMS=np.zeros((number_global_bases,number_global_bases))
            D_VMS_tau=np.zeros((number_global_bases,number_global_bases))

            VMS1=np.zeros((2*number_global_bases,2*number_global_bases))
            VMS2=np.zeros((2*number_global_bases,2*number_global_bases))
            
            ###Code for time dependent bouandry conditions. 
            ###Enforced explicitly or implicitly based on approach used. 
            ###Currently requires incompressability
            
            Sw_in=saturation[0]; Sw_out=saturation[-1]

            Q_VMS=np.zeros(len(self.B)*2)
            Q=np.zeros(len(self.B)*2)
#            
            detJ_in,R_in,dR_dx_in=self.mappings(0,0,np.ones(1)*-1)
            detJ_out,R_out,dR_dx_out=self.mappings(0,self.number_elements-1,np.ones(1))
#           
            dSw_dx_in=np.dot(dR_dx_in[:,0],saturation)
            dSw_dx_out=np.dot(dR_dx_out[:,0],saturation)

            f_w_in=self.F_w(Sw_in);
            f_w_out=self.F_w(Sw_out)

            Sw_norm_in=(Sw_in-self.S_wirr)/(self.S_m-self.S_wirr); Sw_norm_out=(Sw_out-self.S_wirr)/(self.S_m-self.S_wirr)
            eps=.001
            Dc_in=eps*(Sw_norm_in-Sw_norm_in**2); Dc_out=eps*(Sw_norm_out-Sw_norm_out**2)

                    
            q_w_out= -f_w_out*self.source_values_flux[0]; q_o_out= -(1-f_w_out)*self.source_values_flux[0]

            Q[self.source_locations_flux]+=self.source_values_flux/self.phi
#            -self.eps_w*self.k/self.phi*dSw_dx_in
            Q[len(self.B)-1]+=(q_o_out+q_w_out)/self.phi
#            +self.eps_w*self.k/self.phi*dSw_dx_out


            Q[len(self.B)+self.source_locations_flux]+=self.source_values_flux/self.phi-Dc_in*dSw_dx_in          
            Q[-1]+=q_w_out/self.phi+Dc_out*dSw_dx_out
            
#            print(Q[-1])

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
                for bb in range(nelem): 
                    
                    #Finds basis functions supported within element
                    if len(Global_IEN_patch.shape)!=1:
                        global_basis_functions=Global_IEN_patch[bb,:]
                    else:
                        global_basis_functions=Global_IEN_patch
    
                    global_basis_functions=global_basis_functions[0:self.order+1]
                    
                    ###GLobal indeces in element
                    element_indeces=IEN_patch[bb,:]
                    
                    idx_grid = np.ix_(element_indeces,element_indeces)
                 
                    """
                    Solves mapping and NURBS bases at various integration points. Puts results into 2 or 3D array
                    """
                       
                    """
                    Mappings to x,y space
                    """

                    detJ,R,dR_dx=self.mappings(aa,bb,xi_loc,1)
                    dR_dx2=self.mappings(aa,bb,xi_loc,2)

#                    print(dR_dx)
                    """
                    Finding relative permabilities and derivatives of cappilary pressure
                    """                  

                    ###Basis functions
                    R=R[element_indeces,:] #in reverse order now. IEN_patch may be single row
#                    print(R)
                    ###Only taking bais functions with support on current element 
                    dR_dx=dR_dx[element_indeces,:]; dR_dx2=dR_dx2[element_indeces,:];
            
                    ###Finding nodal values of saturation and pressure on element
                    Sw=saturation[element_indeces]; p=pressure[element_indeces];
                    
                    ###Finding sats and pressures at integration points
                    Sw_pts=np.einsum('ij,i->j',R,Sw); p_pts=np.einsum('ij,i->j',R,p);
                    
                    ###Finding relative permeabilties and their derivatives wrt Sw
                    krw,kro=self.relative_permeability(Sw_pts);  dkrw_dSw,dkro_dSw=self.relative_permeability(Sw_pts,1);
    
                    ###Finding mobility and it's derivatve
                    lamw=self.k*krw/self.mu_w/self.phi; dlamw_dSw=self.k*dkrw_dSw/self.mu_w/self.phi; 
                    lamo=self.k*kro/self.mu_o/self.phi; dlamo_dSw=self.k*dkro_dSw/self.mu_o/self.phi; 

                    ###Derivative of pressure with respect to x
                    dp_dx=np.einsum('ij,i->j',dR_dx,p); dp_dx2=np.einsum('ij,i->j',dR_dx2,p);
                    
                    ###Derivative of x wrt Sw
                    dSw_dx=np.einsum('ij,i->j',dR_dx,Sw);dx_dSw=1/dSw_dx; dSw_dx2=np.einsum('ij,i->j',dR_dx2,Sw)
                    
                    ###Fractional flow rates
                    fw=self.F_w(Sw_pts)

                    ###Diffusion coeff
                    Sw_norm=(Sw_pts-self.S_wirr)/(self.S_m-self.S_wirr)
                    eps=.0001
                    Dc=eps*(Sw_norm-Sw_norm**2)
                    dDc_dSw=eps*(self.S_wirr/(self.S_m-self.S_wirr))*(-1+2*Sw_norm**2)
                    
                    ###Element length scale
                    h=self.B[-1,0]/self.number_elements
                    """
                    Calculating pertinent entries in matrices
                    """
                    ###Multiplying bases with lam vec and c vec                 
                    a=np.einsum('ij,j->ij',R,detJ*wl)
                    b=np.einsum('ij,j->ij',dR_dx,lamw*detJ*wl)
                    c=np.einsum('ij,j->ij',dR_dx,lamo*detJ*wl)
#                    d=self.eps_w*self.k/self.phi*np.einsum('ij,j->ij',dR_dx,detJ*wl)
                    d=np.einsum('ij,j->ij',dR_dx,detJ*wl*Dc)
#                    print(d)
                    ###Forming entries of overall partitioned matrices
                    A[idx_grid]+=np.einsum('ji,ki->jk',R,a)
                    B[idx_grid]+=np.einsum('ji,ki->jk',dR_dx,b)          
                    C[idx_grid]+=np.einsum('ji,ki->jk',dR_dx,c)       
                    D[idx_grid]+=np.einsum('ji,ki->jk',dR_dx,d)       

#                    print(xi_loc,element_indeces)
                    if self.SG==1:
                        
#                        tau,L=self.tau(aa,bb,xi_loc,element_indeces,saturation,pressure)
#                        print(L)
                       
                        a1=self.k/self.phi*(dlamw_dSw*dSw_dx*dp_dx+lamw*dp_dx2)
                        a2=dDc_dSw*dSw_dx ###dSW_dx**2???
                        Adv=a1-a2
                        c1=4.; c2=2.;
#                        print(dp_dx)
                        tau=(c1*Dc/h**2+c2*Adv/h)**-1
                        tau=np.nan_to_num(tau)
#                        print(tau)
                    
                        L_1=np.einsum('ij,j->ij',dR_dx,Adv)
                        L_2=np.einsum('ij,j->ij',dR_dx,a2)+np.einsum('ij,j->ij',dR_dx2,Dc)
                        L=-L_1-L_2
                        L=np.nan_to_num(L)
                        
                        
                        a_VMS=np.einsum('ij,j->ij',R,tau*detJ*wl)
#                        b_VMS_tau=np.einsum('ij,j->ij',dR_dx,tau*lam_w_vec*detJ*wl)
#                        c_VMS=np.einsum('ij,j->ij',dR_dx,lam_o_vec*detJ*wl)
#                        d_VMS_tau=self.eps_w*self.k*np.einsum('ij,j->ij',dR_dx,tau*detJ*wl)
                        d_VMS_tau=np.einsum('ij,j->ij',dR_dx,tau*detJ*wl*Dc)

                        A_VMS[idx_grid]+=np.einsum('ji,ki->jk',L,a_VMS)
#                        B_VMS_tau[idx_grid]+=np.einsum('ji,ki->jk',dR_dx,b_VMS_tau)          
#                        C_VMS[idx_grid]+=np.einsum('ji,ki->jk',dR_dx,c_VMS) 
                        D_VMS_tau[idx_grid]+=np.einsum('ji,ki->jk',L,d_VMS_tau)
                        
                        
#                        Q_VMS=np.einsum('i,j->j',tau,Q)

                        """
                        SUPG matrices for no compressability
                        """
#                        K1_VMS=-(C_VMS+B_VMS)
#                        K2_VMS=-B_VMS_tau   
                        K3_VMS=A_VMS/(delta_t)+D_VMS_tau
#                        K4_VMS=D_VMS
                        
                        mat_1=np.hstack((empt_mat,empt_mat))
                        mat_2=np.hstack((empt_mat,K3_VMS))
                        VMS1=np.vstack((mat_1,mat_2))
                        
                        mat_1=np.hstack((empt_mat,empt_mat))
                        mat_2=np.hstack((empt_mat,A_VMS/(delta_t)))
                        VMS2=np.vstack((mat_1,mat_2))

                        
            if mode=='implicit' or mode=='IMPES':
                
#                if self.SG==1:
#                    tau_in,D_in=self.tau(0,0,np.array([-1]),IEN_patch[0,:],saturation,pressure,1)
#                    tau_out,D_out=self.tau(0,-1,np.array([1]),IEN_patch[-1,:],saturation,pressure,1)
#                    Q_VMS=np.zeros(len(Q))
#                    Q_VMS[len(self.B)+IEN_patch[0,:]]=D_in*tau_in*Q[len(self.B)+self.source_locations_flux]
#                    Q_VMS[len(self.B)+IEN_patch[-1,:]]=D_out*tau_out*Q[-1]

                ###Combining common components of partitioned matrices
                K1=-(C+B)
                K2=-B     
                K3=A/(delta_t)+D
                K4=D
#                print(delta_t)
                ###Forming paritioned matrices
#                mat_1=np.hstack((K1,K4))
                mat_1=np.hstack((K1,empt_mat))

                mat_2=np.hstack((K2,K3))
                Kn=np.vstack((mat_1,mat_2))
                
                mat_1=np.hstack((empt_mat,empt_mat))
                mat_2=np.hstack((empt_mat,A/(delta_t)))
                Kn_min1=np.vstack((mat_1,mat_2))
#                print((Kn-VMS1)/Kn)
                return Kn+VMS1,Kn_min1+VMS2,Q,Q_VMS

#            if mode=='explicit':
#                
#                ###Combining common components of partitioned matrices
#                mat_1=E*self.phi/(self.B_w*delta_t)
#                mat_2=np.zeros((number_global_bases,number_global_bases))
#                mat_3=B/delta_t
#                mat_4=A*self.phi/(self.B_w*delta_t)
#                
#                mat_5= -E*self.phi/(self.B_w*delta_t)+C+D*(self.B_o/self.B_w)
#                mat_6=mat_2
#                mat_7=C-B/delta_t
#                mat_8= -A*self.phi/(self.B_w*delta_t)
#                
#                ###Forming paritioned matrices
#                mat_9=np.hstack((mat_1,mat_2))
#                mat_10=np.hstack((mat_3,mat_4))
#                K1=np.vstack((mat_9,mat_10))
#                
#                mat_11=np.hstack((mat_5,mat_6))
#                mat_12=np.hstack((mat_7,mat_8))
#                K2=np.vstack((mat_11,mat_12))
#               
#                return K1,K2 
                                  
    def boundary_conditions(self,saturation,pressure,delta_t,mode='implicit'):
        """
        Applies essential boundary conditions using Lagrange multiplier method.
        ACurrently only applicable to model problem
        """
        """
        Calls assemble to generate matrices
        """
        K1,K2,Q,Q_VMS=self.assemble(saturation,pressure,delta_t,mode=mode)

#        """
#        Applies flux to edge
#        """
#        if self.nuemann_boundaries!=None:
#            for i in range(len(self.nuemann_boundaries)):
#                if self.nuemann_bound[i,0]==-1:
#                    continue
#                direction=self.nuemann_boundaries[i,0]; boundary=self.nuemann_boundaries[i,1]; patch=self.nuemann_boundaries[i,2]
#                flux=self.flux[i]
#                self.edge_integral(direction,boundary,patch,flux)
                

        """
        Applies dirichlet conditions to points. Assuming that stauration must also be specified at inlet
        """    
        if self.source_locations_pressure!=None:

            Q+= -np.einsum('ij,j',K1[:, self.source_locations_pressure],self.source_values_pressure)
            K1[:,self.source_locations_pressure]=0
            K1[self.source_locations_pressure,:]=0
            K1[self.source_locations_pressure,self.source_locations_pressure]=1.
#            print(Q)
            Q+= np.einsum('ij,j',K2[:, self.source_locations_pressure],self.source_values_pressure)      
            K2[:,self.source_locations_pressure]=0
            K2[self.source_locations_pressure,:]=0
        
            Q[self.source_locations_pressure]=self.source_values_pressure         
            Q_VMS[self.source_locations_pressure]=0
            
        if self.boundary_saturations_locations!=None:
#             
            Q+= -np.einsum('ij,j',K1[:, len(self.B)+self.boundary_saturations_locations],self.boundary_saturations_values)
#            print(K2)
            K1[:,len(self.B)+self.boundary_saturations_locations]=0
            K1[ len(self.B)+self.boundary_saturations_locations,:]=0
            K1[len(self.B)+self.boundary_saturations_locations,len(self.B)+self.boundary_saturations_locations]=1.
            
            Q+= np.einsum('ij,j',K2[:, len(self.B)+self.boundary_saturations_locations],self.boundary_saturations_values)
            K2[:,len(self.B)+self.boundary_saturations_locations]=0
            K2[ len(self.B)+self.boundary_saturations_locations,:]=0
            
            Q[ len(self.B)+self.boundary_saturations_locations]=self.boundary_saturations_values  
            Q_VMS[len(self.B)+self.boundary_saturations_locations]=0
        return K1,K2,Q,Q_VMS
    
    def residuals(self,nodal_vals_vec=None,mode='implicit'):
        """
        Finds residuals
        
        Use inital saturation for explicit solver, current satration for implicit        

        """

        pressure_init=self.pressure_init; saturation_init=self.saturation_init
        delta_t=self.delta_t
    
        """
        Explicit solver
        """

        if mode=='explicit':
            K1,K2,Q=self.boundary_conditions(saturation_init,delta_t,mode=mode)
#            print K1
#            print K2
#            print Q
            return np.linalg.solve(K1,np.einsum('ij,j->i',-K2,np.hstack((pressure_init,saturation_init)))+Q)
             
        """
        Implicit solver
        """

        if mode=='implicit':
            pressure=nodal_vals_vec[0:len(self.B)]; saturation=nodal_vals_vec[len(self.B):]
            K1,K2,Q=self.boundary_conditions(saturation,delta_t,mode=mode)
            R=np.einsum('ij,j->i',K1,np.hstack((pressure,saturation)))-np.einsum('ij,j->i',K2,np.hstack((pressure_init,saturation_init)))-Q      
#            print saturation_init 
#            print K1
            return R
        
    def IMPES(self,pressure):
        """
        IMPES solver
        """
        pressure_init=self.pressure_init; saturation_init=self.saturation_init
        delta_t=self.delta_t

        K1,K2,Q,Q_VMS=self.boundary_conditions(saturation_init,pressure_init,delta_t,mode='IMPES')


        K1_p=K1[:len(self.B),:len(self.B)]
        K2_p=K2[:len(self.B),:len(self.B)]
        Q=Q[:len(self.B)]
        Q_VMS=Q_VMS[:len(self.B)]
#        R=np.einsum('ij,j->i',K1_p,pressure)-np.einsum('ij,j->i',K2_p,pressure_init)-Q-Q_VMS
        R=np.linalg.solve(K1_p,np.einsum('ij,j->i',K2_p,pressure_init)+Q+Q_VMS)      
#        print (Q,Q_VMS)
        return R
    
    
    def create_solution(self,mode='implicit'):
        """
        Loops over times and uses Newton-Kryloft method for solving nonlinear set of equations
        """
       
        pressures_and_saturations=np.zeros((2*len(self.B),len(self.times)))
#        print pressures_and_saturations,self.B
#        self.initial_saturation[1]=self.S_m
        pressures_and_saturations[:,0]=np.hstack((self.initial_pressure,self.initial_saturation))
#        print  pressures_and_saturations[:,0]
        for time in range(1,len(self.times)):
            self.delta_t=self.times[time]-self.times[time-1]
#            print(self.delta_t,self.times[time])
            self.pressure_init=pressures_and_saturations[0:len(self.B),time-1]
            self.pressure_init[self.source_locations_pressure]=self.source_values_pressure

            self.saturation_init=pressures_and_saturations[len(self.B):,time-1]
#            if time==1:
#                self.saturation_init[1]=1
            ###Line for running implicit solver
            if mode=='implicit':
                 p_and_s=newton_krylov(self.residuals,np.hstack((self.pressure_init,self.saturation_init)))
#                 p_and_s=newton_krylov(self.residuals,np.hstack((self.pressure_init,np.ones(len(self.B))*.5)))

            ###Line for running explicit solver
            elif mode=='explicit':
                p_and_s=self.residuals(mode='explicit')
            
            elif mode=='IMPES':
                s_iter=self.saturation_init
#                p_current=newton_krylov(self.IMPES,self.pressure_init)
#                print(self.pressure_init)
                p_current=self.IMPES(self.pressure_init)
                K1,K2,Q,Q_VMS=self.boundary_conditions(self.saturation_init,p_current,self.delta_t,mode='IMPES')
                K1_p=K1[len(self.B):,:len(self.B)]; K1_s=K1[len(self.B):,len(self.B):]
                K2_p=K2[len(self.B):,:len(self.B)]; K2_s=K2[len(self.B):,len(self.B):]
                Q=Q[len(self.B):]
                Q_VMS=Q_VMS[len(self.B):]
                s_current=np.linalg.solve(K1_s,-np.einsum('ij,j->i',K1_p,p_current)+np.einsum('ij,j->i',K2_p,self.pressure_init)+np.einsum('ij,j->i',K2_s,self.saturation_init)+Q+Q_VMS)
#                s_current = np.where(s_current<self.S_wirr,self.S_wirr,s_current)                
                count=0
#                while np.sqrt(np.sum((s_iter-s_current)**2))>10**-3:
                while count<2:
#
                    s_iter=s_current
                    p_iter=p_current
                    self.saturation_init=s_iter
                    self.pressure_init=p_iter

                    p_current=self.IMPES(self.pressure_init)

#                    p_current=newton_krylov(self.IMPES,self.pressure_init)
                    K1,K2,Q,Q_VMS=self.boundary_conditions(self.saturation_init,self.pressure_init,self.delta_t,mode='IMPES')
                    K1_p=K1[len(self.B):,:len(self.B)]; K1_s=K1[len(self.B):,len(self.B):]
                    K2_p=K2[len(self.B):,:len(self.B)]; K2_s=K2[len(self.B):,len(self.B):]
                    Q=Q[len(self.B):]
                    Q_VMS=Q_VMS[len(self.B):]

                    self.saturation_init=pressures_and_saturations[len(self.B):,time-1]
                    s_current=np.linalg.solve(K1_s,-np.einsum('ij,j->i',K1_p,p_current)+np.einsum('ij,j->i',K2_p,self.pressure_init)+np.einsum('ij,j->i',K2_s,self.saturation_init)+Q+Q_VMS)
#                    s_current = np.where(s_current<self.S_wirr,self.S_wirr,s_current)                
                    
                    count+=1

                p_and_s=np.hstack((p_current,s_current))

            pressures_and_saturations[:,time]=p_and_s

        self.pressures=pressures_and_saturations[0:len(self.B),:]
        self.saturations=pressures_and_saturations[len(self.B):,:]
        return
           
    def approx_function(self,patch_num,xi,time_step,output='pressures'):
        """
        Finds solution at given xi_til,eta_til, and patch number
        """
        if output=='pressures':
            approx=self.pressures[:,time_step]  
        elif output=='saturations':
            approx=self.saturations[:,time_step]
        refs=self.patch_data.patch_global_2D(patch_num)
        refs=refs.flatten()

        patch_bases=self.NURBS_bases(patch_num,xi)
     
        return np.dot(approx,patch_bases)
    
    def make_frame_p(self,i,num_pts=200):
        """
        Trying to animate in window
        """
#        xi_pts=np.linspace(0,1,num_pts)
#        pressures=self.approx_function(0,xi_pts,i,'pressures')
##        plt.plot(xi_pts*max(self.B[:,0]),pressures)
#        self.line.set_data(xi_pts*max(self.B[:,0]), pressures)
#        return self.line,
        """
        Animating by saving collection of figures
        """
        for j in range(len(self.times)):
            fig, ax = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
            ax.set_ylim(np.min(self.pressures), np.max(self.pressures))
            ax.set_xlim(min(self.B[:,0]),max(self.B[:,0]))
            xi_pts=np.linspace(0,1,num_pts)
            pressures=self.approx_function(0,xi_pts,j,'pressures')
            ax.plot(xi_pts*max(self.B[:,0]),pressures)
            plt.xlabel('x')
            plt.ylabel('Pressure')
            text(.8,.9,'time=%.3f' % self.times[j],ha='center', va='center',transform=ax.transAxes)
            file_name=('pressure_image%d.png' % j)
            fig.savefig('./pressure_images/%s' % file_name)

    def L2_error_Sw(self,i,num_pts=200):
        xi_pts=np.linspace(0,1,num_pts)
        x1=xi_pts*max(self.B[:,0]); y1=self.approx_function(0,xi_pts,i,'saturations')
        x_BL,Sw_BL=Buckley_Leverett(np.array([self.mu_w,self.mu_o]),np.array([self.phi,self.S_wirr,self.S_m]),self.source_values_flux[0],1,self.k[0],self.B[-1,0],self.times[i]).sat_curve(num_pts)
        L2=np.sqrt(np.sum((Sw_BL-y1)**2))
        dof=len(self.B)*2
        return dof,L2
        
    def anim_saturation(self,filename):
#        fig, ax = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
        fig=plt.figure()
        ax=plt.axes()
        ax.set_ylim(0, 1)
        ax.set_xlim(min(self.B[:,0]),max(self.B[:,0]))
        plt.xlabel('Position [m]')
        plt.ylabel('S_W')
        plt.grid('on')
        approx,=ax.plot([],[],label='Approximation')
        BL,=ax.plot([],[],label='Buck. Leverett')
        ax.legend(handles=[approx,BL],loc=1)
        txt=plt.text(.8,.1,'init',ha='center', va='center',transform=ax.transAxes)

        def animate_init():
            approx.set_data([],[])
            BL.set_data([],[])
            return approx,BL,
        def animate_saturation(i,num_points=200):
            txt.set_text('time=%.3f' % self.times[i])
            plt.show
            xi_pts=np.linspace(0,1,num_points)
            x1=xi_pts*max(self.B[:,0]); y1=self.approx_function(0,xi_pts,i,'saturations')
            if i!=0:
                x_BL,Sw_BL=Buckley_Leverett(np.array([self.mu_w,self.mu_o]),np.array([self.phi,self.S_wirr,self.S_m]),self.rel_perm_params,self.source_values_flux[0],1,self.k[0],self.B[-1,0],self.times[i]).sat_curve(num_points)
                BL.set_data([x_BL],[Sw_BL])
          
            approx.set_data(x1,y1)
            return approx,BL,
        
#        init_func=self.animate_init,
        anim = animation.FuncAnimation(fig,animate_saturation, init_func=animate_init,frames=np.arange(len(self.times)), interval=100, blit=True)
        anim.save(filename)
#        writer='mencoder'
        plt.show()
    def make_frame_s(self,i,num_pts=200,BL=False):
#        xi_pts=np.linspace(0,1,num_pts)
#        saturations=self.approx_function(0,xi_pts,i,'saturations')
##        plt.plot(xi_pts*max(self.B[:,0]),pressures)
#        self.line.set_data(xi_pts*max(self.B[:,0]), saturations)
#        return self.line
        """
        Animating by saving collection of figures
        """
        for j in range(len(self.times)):
            fig, ax = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
            ax.set_ylim(0, 1)
            ax.set_xlim(min(self.B[:,0]),max(self.B[:,0]))
            xi_pts=np.linspace(0,1,num_pts)
            saturations=self.approx_function(0,xi_pts,j,'saturations')
            ax.plot(xi_pts*max(self.B[:,0]),saturations)
            if j!=0:
                x_BL,Sw_BL=Buckley_Leverett(np.array([self.mu_w,self.mu_o]),np.array([self.phi,self.S_wirr,self.S_m]),self.rel_perm_params,self.source_values_flux[0],1,self.k[0],self.B[-1,0],self.times[j]).sat_curve()
                ax,plot(x_BL,Sw_BL)
            plt.xlabel('x')
            plt.ylabel('Saturation')
            text(.8,.9,'time=%.3f' % self.times[j],ha='center', va='center',transform=ax.transAxes)
            text(.8,.8,'avg=%.3f' % np.average(saturations),ha='center', va='center',transform=ax.transAxes)
           
            file_name=('saturation_image%d.png' % j)
            fig.savefig('./saturation_images/%s' % file_name)
            
    def plotter(self,output='pressure'):       
        """
        Plots solution by interpolating between nodes
        """  
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)

        self.line, = self.ax.plot([], [], 'bo', ms=2)
#        plt.show()
        if output=='pressure':
            self.ax.set_ylim(np.min(self.pressures), np.max(self.pressures))
            self.ax.set_xlim(min(self.B[:,0]),max(self.B[:,0]))
            ani = animation.FuncAnimation(self.fig, self.make_frame_p, frames=range(len(self.times)),\
            interval=1, repeat=True,blit=True)  
            return ani
#            for data_file in os.listdir('./pressure_images'):
##                a=Image.open('./pressure_images/%s' % data_file)
#                a=open('./pressure_images/%s' % data_file)
#                time.sleep(.5)
#                a.close()
        if output=='saturation':
            ax.set_ylim(np.min(self.saturations), np.max(self.saturations))
            ax.set_xlim(min(self.B[:,0]),max(self.B[:,0]))
            ani = animation.FuncAnimation(fig, self.make_frame_s, frames=len(self.times),\
            interval=1, repeat=True) 
       
        
       
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


