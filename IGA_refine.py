
"""
Code for refining IGA knot vector and adding in control point locations

Created on Sat Jul 18 21:22:04 2015

@author: Eric
"""
import numpy as np
import math
from IGA_pre_processing import patch_data
import matplotlib.pyplot as plt

def knot_vector(number_elements,order,multiplicity):
        """
        Computes knot vector assuming open, uniform vector. Currently uses multiplicity multiplicity consistent with C_0 elements 
        """
        knot_vector_interior=np.linspace(0,1,number_elements+1)
        knot_vector_interior=np.repeat(knot_vector_interior[1:-1],multiplicity) 
        
        #Add zeros to beginning
        knot_vector=np.append(np.zeros(order+1),knot_vector_interior)
        
        #Add ones to end
        knot_vector=np.append(knot_vector,np.ones(order+1))
        
        return knot_vector
        
class IGA_h_refine():
    """
    Performs h refinement over a 1-D curve. Outputs new control net. See Hughes'
    book for algorithm
    """
    
    def __init__(self,KV,KV_new,B_old,n,n_new,p):
         """
         INPUTS:
         
         KV:     Original knot vector
         KV_new: New knot vector
         B_old:  Control net for original knot vector
         n:      Number of original NURBS bases
         n_new:  Number of added NURBS bases (n+n_new=total number new BASES)
         p:      NURBS order
         """
         
         """
         Initialization of terms
         """
         
         self.KV=KV
         self.KV_new=KV_new
        
         """
         Transform NURBS to B-spline 
         """
         weights=B_old[:,-1]
         self.B_old=B_old
         self.B_old[:,0:-1]=B_old[:,0:-1]*weights[:,None]
         
         
         
         self.n=n
         self.n_new=n_new
         self.p=p
         
         """
         Save new control net. Transform back to NURBS
         """
         self.B_new=self.h_refine_KV()[:,:]
         weights_new= self.B_new[:,-1]
         self.B_new[:,0:-1]=self.B_new[:,0:-1]/weights_new[:,None]


    def T_0(self,KV_new,KV_old,i,j):
        """
        Lowest level of recursive algorithm
        
        INPUTS:
        KV_new:     New knot vector
        KV_old:     Old knot vector
        i:          Index variable
        j:          Index variable
        """
        
        return 1. if ((KV_new[i]>= KV_old[j] and KV_new[i]<KV_old[j+1])) else 0 #ask for function that check truth value if number in a range
#        or (KV_new[i]== KV_old[j])) 
    
    def T_ij(self,KV_new,KV_old,q,i,j):
        """
        Uses recursive function to solve for each entry T_ij in T. T is a transform 
        matrix that converts old control net to new.
        
        INPUTS:
        KV_new:      New knot vector
        KV_old:      Old knot vector
        q:           Gives level of recursion
        i:           Index variable
        j:           Index variable
        """ 
        
        if q==0:
            return self.T_0(KV_new,KV_old,i,j)
        else:
            T_ij_minus1=self.T_ij(KV_new,KV_old,q-1,i,j)
            T_ij_jplus1=self.T_ij(KV_new,KV_old,q-1,i,j+1)
        
        """
        Creates numerators and denominators
        """
        num1=KV_new[i+q]-KV_old[j]
        dem1=KV_old[j+q]-KV_old[j]
         
        num2=KV_old[j+q+1]-KV_new[i+q] #need to see if this is an error
        dem2=KV_old[j+q+1]-KV_old[j+1]
        
        """
        Handles divide-by-zero errors
        """
        with np.errstate(divide='ignore', invalid='ignore'):
            
            first_term = np.where(dem1 != 0.0, 
                                  (num1 / dem1), 0.0)
            second_term = np.where(dem2 != 0.0,
                                   (num2 /dem2), 0.0)
       
#        first_term=num1/dem1; second_term=num2/dem2                             
        return first_term*T_ij_minus1+second_term*T_ij_jplus1
            
    def create_T(self,KV_new,KV_old,old_num_bases,new_bases,order):  
        """
        Uses T_ij to generate each entry of T
        
        INPUTS:
        KV_new:        New knot vector
        KV_old:        Old knot vector
        old_num_bases: Total number of original bases 
        new_bases:     Number of added bases
        order:         Order of bases
        """
        rows=old_num_bases+new_bases
        cols=old_num_bases
        T=np.zeros((rows,cols))
        
        """
        Iterates through each entry of T_ij
        """
        for i in range(rows):
            for j in range(cols):
                T[i,j]=self.T_ij(KV_new,KV_old,order,i,j) #how do you reference current indeces?
                
        return T
    
    def h_refine_KV(self):
        """
        Refine single knot vector using T matrix
        
        Outputs new control new
        """
        
        """
        Initialize new control net
        """
        B_new=np.zeros((self.n+self.n_new,3))
        
        """
        Run T creation algorithm
        """
        T=self.create_T(self.KV_new,self.KV,self.n,self.n_new,self.p)
#        return T,self.B_old
#        print(T)
        for row in range(self.n+self.n_new):
            for col in range(self.n):
                B_new[row,:]+=self.B_old[col,:]*T[row,col]
            
        return B_new
        
        
class IGA_knot_removal():
    """
    Reduces knot multiplicty uniformly. Currently does not have check to see 
    if knot is actually removable. See pg. 185 in Piegl and Tiller for reference
    """
    def __init__(self,p,KV_old,B_old,num):
        
        """
        INPUTS:
        
        p:      NURBS order
        KV_old: Old knot vector
        B_old:  Old control net
        num:    Number of times the internal knots will have multiplicity reduced
        """
        self.p=p #Polynomial order
        self.KV_old=KV_old #knot vector
        
        
#        self.B_old=B_old #Original control net
        """
        Transform NURBS to B-spline 
        """
        weights=B_old[:,-1]
        self.B_old=B_old[:,:]
        self.B_old[:,0:-1]=B_old[:,0:-1]*weights[:,None]
        
        
        
        
        self.num=num #Number of attempts with which the knot will be removed
        
#        self.n=p-s+1 #Knot removal destroys n control points and replaces them with n-1 new control points
        
        self.B_new,self.KV_new=self.reduce_order_KV()
        
        
        weights_new= self.B_new[:,-1]
        self.B_new[:,0:-1]=self.B_new[:,0:-1]/weights_new[:,None]
        
        
    def unique_rows(self,data):
        """
        Finds unique rows in a 2D array. Found at:
        
        http://stackoverflow.com/questions/16970982/find-unique-rows-in-numpy-array
        """
        uniq = np.unique(data.view(data.dtype.descr * data.shape[1]))
        return uniq.view(data.dtype).reshape(-1, data.shape[1])

    
    def remove_knot(self,u,r,s,B_old,KV_old):
        """
        Reduces multiplicity of one knot
        
        INPUTS:
        u:     Knot value to be removed
        r:     Python index of right most knot value to be removed
        s:     Multplicity of knot to be removed
        B_old: Old control net
        KV_old:Old knot vector
        """
        
        p=self.p
        
        """
        Initial value for i and j index variables
        """
        i_init=r-p
        j_init=r-s
        
        """
        Follows algorithm in NURBS book, Piegl and Tiller
        """
        
#        B_minus1=np.zeros([len(B_old),3]) #Initialize B for current loop 
#        B_minus1[:]=B_old[:] #Initialize B_minus1 if t=1
        
        
#        rows=np.shape(B_minus1)[0]
#        cols=np.shape(B_minus1)[1]
             
        
#        B=np.zeros([rows,cols]) #Initialize B for current loop 
        B=B_old
        
        
        for t in np.arange(0,self.num):
            B_minus1=B
#            B=np.zeros([rows-1,cols]) #Initialize B for current loop
#            B[0,:]=B_minus1[0,:]; B[-1,:]=B_minus1[-1,:]
#            rows+=-1
            i_start=r-p-t+1 #Initializing i and j for each iteration
            j_start=r-s+t-1
            
#            return i_start,j_start
#            if t==1:
              
            i=i_start
            j=j_start
            

            #Applies theorem 5.28
            while j-i>t-1:
                alpha_i=(u-KV_old[i])/(KV_old[i+p+t]-KV_old[i])
                alpha_j=(u-KV_old[j-t+1])/(KV_old[j+p+1]-KV_old[j-t+1])

                B[i]=(B_minus1[i]-(1-alpha_i)*B[i-1])/alpha_i
                B[j]=(B_minus1[j]-alpha_j*B[j+1])/(1-alpha_j)
                
                i+=1; j+= -1    
             
            
        
#        if np.array_equal(B[i],B[j]):
#            B_new=np.vstack((B[0:i_start,:],B[j_start:-1,:]))
#        else:
#            B_new=np.vstack((B[0:i_start+1,:],B[j_start:-1,:]))
        """
        Output new KV
        """
        indeces=np.where(KV_old==u)[0][:self.num] #selects entries to be removed from knot vector
        KV_new=np.delete(KV_old,indeces)
#        print(np.any(np.where(B==B_minus1)))
#        print(B)
#        return B,KV_new
        """
        Output new control net. Deviates from book's algorithm.
        """
#        return B,i_init,j_init

        if i_init==j_init: #Condition if number of rows is odd
#            return np.allclose(B[i_init-(self.num-1)],B[j_init+(self.num-1)],10**-10,0)
        
            if self.num%2==0:
                B_new=np.delete(B,range(int(i_init-self.num/2.),int(i_init+self.num/2.)),axis=0)
                B_new[i_init-self.num/2]=np.average([B[i_init-self.num/2.],B[i_init+self.num/2.]],axis=0) #THIS MAY NOT WORK FOR GENERAL CASE
#                B_new=self.unique_rows(B_new)
            else:
                B_new=np.delete(B,range(int(i_init-(self.num+1)/2.+1),int(i_init+(self.num+1)/2.)),axis=0)
#            if self.num==1:
#                B_new=np.delete(B,i_init,axis=0)
#            elif np.allclose(B[i_init-(self.num-1)],B[j_init+(self.num-1)],10**-10,0)==0 and self.num>1:
#                B_new=np.vstack((B[0:i_init-(self.num-1)+1,:],B[j_init+(self.num-1):,:]))
#            elif np.allclose(B[i_init-(self.num-1)],B[j_init+(self.num-1)],10**-10,0)==1 and self.num>1:
#                B_new=np.vstack((B[0:i_init-(self.num-1),:],B[j_init+(self.num-1):,:]))
        else: #condition if number of rows is even
            
            if self.num==1:
                B_new=self.unique_rows(B)
            elif self.num%2==0:
                B_new=np.delete(B,range(int(i_init-self.num/2.+1),int(j_init+self.num/2.)),axis=0)
            elif self.num%2!=0:
                B_new=np.delete(B,range(int(i_init-(self.num-1)/2.),int(j_init+(self.num-1)/2.)),axis=0)
#                B_new=self.unique_rows(B)
#            if self.num==1:
#                B_new=B[:]
#            elif np.allclose(B[i_init-(self.num-1)],B[j_init+(self.num-1)],10**-10,0)==0 and self.num>1:
#                B_new=np.vstack((B[0:i_init-(self.num-1)+1,:],B[j_init+(self.num-1):,:]))
#            elif np.allclose(B[i_init-(self.num-1)],B[j_init+(self.num-1)],10**-10,0)==1 and self.num>1:
#                B_new=np.vstack((B[0:i_init-(self.num-1),:],B[j_init+(self.num-1):,:]))
        
#        print(B_new`)
        return B_new,KV_new #returns b spline
    
    def reduce_order_KV(self):
        """
        Reduces multiplicity of all internal knots in a uniform knot vector. s and num assumed to be constant for each
        knot value
        """
        KV_old=self.KV_old
        
        """
        Finds indeces and values of interal knots
        """
        inner_knot_inds=np.where(np.logical_and(KV_old!=0,KV_old!=1))
        inner_knot_vals=np.unique(KV_old[inner_knot_inds])
        B_current=self.B_old[:,:] #Initialize KV and Bspline B
        KV_current=KV_old[:]
        
        """
        Iterates through each internal knot using remove_knot
        """
        for i in range(len(inner_knot_vals)):
            u_current=inner_knot_vals[i]
            r_current=np.max(np.where(KV_current==u_current))
            s_current=len(np.where(KV_current==u_current)[0])

#            if i==1:
#                return B_current,KV_current,u_current,s_current,r_current
                
            B_current,KV_current=self.remove_knot(u_current,r_current,s_current,B_current,KV_current)
#            if i==2:
#                return B_current,KV_current,u_current,s_current,r_current
#        if i==0:
#            return B_current,KV_current,u_current,s_current,r_current
        B_new=B_current[:]
        KV_new=KV_current[:]
#        if i==0:
#            return B_current,KV_current,u_current,s_current,r_current
        return B_new,KV_new
   

     
class IGA_p_refine():
    """
    Performs p refinement on single patch
    """
    
    def __init__(self,p,p_new,B_old,KV_old):
        """
        INPUTS:
        
        p:     Original polynomial order
        p_new: New polynomial order
        B_old: Old control net
        KV_old:Old knot vector
        """
        
        """
        Initialize
        """
        self.p=p
        self.p_new=p_new
        self.t=p_new-p
        self.NURBS_B_old=B_old
        
#        weights=B_old[:,-1]
#        self.Bspline_B_old[:,0:-1]=B_old[:,0:-1]*weights[:,None]
        
        self.KV_old=KV_old
        
        """
        Save new control net and knot vector
        """
        self.B_new,self.KV_new=self.p_refine_KV()
       
    def bin_coeff(self,x,y):
        """
        Finds a binomial coefficient. Found at http://stackoverflow.com/questions/26560726/python-binomial-coefficient
        """
        
        if y == x:
            return 1.
        elif y == 1:         # see georg's comment
            return float(x)
        elif y > x:          # will be executed only if y != 1 and y != x
            return 0
        else:                # will be executed only if y != 1 and y != x and x <= y
            a = float(math.factorial(x))
            b = float(math.factorial(y))
            c = float(math.factorial(x-y)) # that appears to be useful to get the correct result
            div = a/(b * c)
            return div
            
    
            
    def p_refine_Bezier(self,B_old):
        """
        Finds new control points after t order elevations of single Bezier curve. Refines knot up to 
        C_0 continuity.
        See pg. 205 in "The NURBS book"
        
        INPUTS:
        B_old: Old control net
        """      
        """
        Convert to Bsplines
        """
        weights=B_old[:,-1]
        B_old[:,0:-1]=B_old[:,0:-1]*weights[:,None]
        
        for i in range(self.p_new+1):
            start_j=max(0,i-self.t)
            end_j=min(self.p,i)
            B_new_i=np.zeros(3)
            for j in range(start_j,end_j+1): #may need to add 1
                B_new_i+=self.bin_coeff(self.p,j)*self.bin_coeff(self.t,i-j)*B_old[j] / self.bin_coeff(self.p+self.t,i)

            if i==0:
                B_new=B_new_i[:]
            else:
                B_new=np.vstack((B_new,B_new_i))
        
        """
        Convert back to NURBS
        """
        weights_new= B_new[:,-1]
        B_new[:,0:-1]=B_new[:,0:-1]/weights_new[:,None]
        return B_new
        
    def h_refine_C0(self):
        """
        Refine knot vector such that it is compatible with p-refinement scheme. Raises mult.
        of inner knots s.t. bases are Co continuous across elements
        """
        """
        Initialize B_old and KV_old
        """
        B_old=self.NURBS_B_old[:]
        KV_old=self.KV_old[:]
        
        """
        Find indeces,values, and multplicities of inner knots
        """
        inner_inds=np.where(np.logical_and(KV_old!=0,KV_old!=1))
        inner_vals=self.KV_old[inner_inds]
        unique_inner_vals=np.unique(inner_vals) #Values of inner knots
        
        """
        Insert knots into original knot vector until multplicity of inner knots=p_old. 
        Run h_refine to get intermediate control nets
        """
        
        """
        Gathers info for each internal knot
        """
        for i in range(len(unique_inner_vals)):
            inner_knot_val=unique_inner_vals[i]
            current_inner_val_inds=np.where(KV_old==inner_knot_val) 
         
            """
            Runs current knot through h refinement
            """
            if len(current_inner_val_inds[0])<self.p:
                num_added_knots=self.p-len(current_inner_val_inds)
                added_knot_array=np.ones(num_added_knots)*inner_knot_val
                insert_start=np.max(current_inner_val_inds)+1
                insert_indeces=np.ones(num_added_knots)*insert_start
                KV_new=np.insert(KV_old,insert_indeces,added_knot_array)

                """
                Run h refine
                """
                n_current=len(KV_old)-(self.p+1)
                n_new_current=num_added_knots
                current_h_refine=IGA_h_refine(KV_old,KV_new,B_old,n_current,n_new_current,self.p)

                """
                Re define B_old as recently refine control net. Same with KV
                """
                B_old=current_h_refine.B_new
                KV_old=current_h_refine.KV_new

            else:
                continue
            
        """
        B_old= newest control net in final iteration
        """
        return B_old,KV_old
        
    def p_refine_KV(self):
         """
         Raises polynomial order over various Bezier segments
         """
         B_old,KV_old=self.h_refine_C0()
         num_elements=len(np.unique(KV_old))-1
         KV_new=knot_vector(num_elements,self.p_new,self.p_new)
         
         B_new=B_old[0:self.p_new]
         
         for ele_num in range(num_elements):
             B_current=B_old[self.p*ele_num:self.p*ele_num+self.p+1]
#             return B_current
             if ele_num==0:
                 B_new=self.p_refine_Bezier(B_current)
#                 return B_new,B_current
             else:
                 B_new_i=self.p_refine_Bezier(B_current)
                 B_new=np.vstack((B_new,B_new_i[1:]))
#                 return B_new,B_current,B_new_i
                 
         return B_new,KV_new

class Line_refine():
    """
    Performs h,p, and/or k refinement over a single line. 
    """
    def __init__(self,patch_data_old,patch_data_new,B_old):
        
        """
        INPUTS:
        
        mult:     Array containing data for knot multiplicity
        num_ele:  Array containing data for number of elements
        order:    Array containing data for basis order
        B_old:    Original connectivity array
            
        1st column: Orginal info
        2nd column: New info
        """
        self.mult_old=patch_data_old.mp; self.mult_new=patch_data_new.mp
        self.num_ele_old=patch_data_old.number_elements; self.num_ele_new=patch_data_new.number_elements
        self.p_old=patch_data_old.order; self.p_new=patch_data_new.order

        self.KV_old=knot_vector(self.num_ele_old,self.p_old,self.mult_old)
        self.KV_new=knot_vector(self.num_ele_new,self.p_new,self.mult_new)
                
        self.B_old=B_old
        
        """
        Number of old and new basis functions 
        """
        self.n_old=len(self.KV_old)-(self.p_old+1)
        self.n_new=len(self.KV_new)-(self.p_new+1)

        
        self.B_new=self.refine_line()
        
    def refine_line(self):
        """
        Refines NURBS surface
        """
        
        """
        Assumes proper ordering of control points in B_old
        """
#        print(self.m_old,self.n_old,self.B_old[:,0])
        B_x_old=self.B_old[:,0]
        B_y_old=self.B_old[:,1]
        B_w_old=self.B_old[:,2]

        """
        Refines in xi direction 
        """
        B_x_current=B_x_old
        B_y_current=B_y_old 
        B_w_current=B_w_old 

        B_current=np.vstack((B_x_current,B_y_current,B_w_current)).T
        
        KV_current=self.KV_old[:]
        
        """
        Runs p refine first
        """
        if self.p_new>self.p_old: #p_new must be greater than or equal to p_old
            p_refine=IGA_p_refine(self.p_old,self.p_new,B_current,KV_current)
            B_current=p_refine.B_new
            KV_current=p_refine.KV_new
            
        """
        If necessary, reduces multiplicity
        """
        if self.mult_new<self.p_new: #p_new is max multipliity of new knots
            t= self.p_new-self.mult_new
            knot_remove=IGA_knot_removal(self.p_new,KV_current,B_current,t)
            B_current=knot_remove.B_new
            KV_current=knot_remove.KV_new
            
        """
        Performs h refinement if neccesary
        """
        if self.num_ele_new>self.num_ele_old: #p_refine will always output C_0 continuous scheme
            n_current=len(KV_current)-(self.p_new+1)
            n_added=len(self.KV_new)-len(KV_current)
            
            h_refine= IGA_h_refine(KV_current,self.KV_new,B_current,n_current,n_added,self.p_new)
            B_current=h_refine.B_new
            KV_current=h_refine.KV_new
        
        
        return B_current

    


class Surface_refine():
    """
    Performs h,p, and/or k refinement over a single patch. Will be modified later for multiple patches
    """
    def __init__(self,mult,num_ele,order,B_old):
        
        """
        INPUTS:
        
        mult:     Array containing data for knot multiplicities
        num_ele:  Array containing data for number of elements
        order:    Array containing data for basis orders
        B_old:    Original connectivity array
        
        Array structures are as follows:
        
        1st row: xi info
        2nd row: eta info
        
        1st column: Orginal info
        2nd column: New info
        """
        self.mult_xi_old=mult[0,0]; self.mult_xi_new=mult[0,1]; self.mult_eta_old=mult[1,0]; self.mult_eta_new=mult[1,1]
        self.num_ele_xi_old=num_ele[0,0]; self.num_ele_xi_new=num_ele[0,1]; self.num_ele_eta_old=num_ele[1,0]; self.num_ele_eta_new=num_ele[1,1]
        self.p_old=order[0,0]; self.p_new=order[0,1]; self.q_old=order[1,0]; self.q_new=order[1,1] 

        self.KV_xi_old=knot_vector(self.num_ele_xi_old,self.p_old,self.mult_xi_old)
        self.KV_xi_new=knot_vector(self.num_ele_xi_new,self.p_new,self.mult_xi_new)
        self.KV_eta_old=knot_vector(self.num_ele_eta_old,self.q_old,self.mult_eta_old)
        self.KV_eta_new=knot_vector(self.num_ele_eta_new,self.q_new,self.mult_eta_new)
                
        self.B_old=B_old
        
        """
        Number of old and new basis functions in xi and eta directions
        """
        self.n_old=len(self.KV_xi_old)-(self.p_old+1)
        self.n_new=len(self.KV_xi_new)-(self.p_new+1)
        self.m_old=len(self.KV_eta_old)-(self.q_old+1)
        self.m_new=len(self.KV_eta_new)-(self.q_new+1)
        
        self.B_new=self.refine_surface()
        
    def refine_surface(self):
        """
        Refines NURBS surface
        """
        
        """
        Number of control points equals number of basis functions
        """
        B_new_x=np.zeros((max(self.m_new,self.m_old),max(self.n_new,self.n_old)))
        B_new_y=np.zeros((max(self.m_new,self.m_old),max(self.n_new,self.n_old)))
        B_new_w=np.zeros((max(self.m_new,self.m_old),max(self.n_new,self.n_old)))
        
        """
        Assumes proper ordering of control points in B_old
        """
#        print(self.m_old,self.n_old,self.B_old[:,0])
        B_x_old=self.B_old[:,0].reshape(self.m_old,self.n_old)
        B_y_old=self.B_old[:,1].reshape(self.m_old,self.n_old)
        B_w_old=self.B_old[:,2].reshape(self.m_old,self.n_old)

        """
        Refines in xi direction first
        """
        for row in range(self.m_old):
            B_x_current=B_x_old[row,:]
            B_y_current=B_y_old[row,:]
            B_w_current=B_w_old[row,:]
            

            B_current=np.vstack((B_x_current,B_y_current,B_w_current)).T
            
            KV_current=self.KV_xi_old[:]
            
            """
            Runs p refine first
            """
            if self.p_new>self.p_old: #p_new must be greater than or equal to p_old
                p_refine=IGA_p_refine(self.p_old,self.p_new,B_current,KV_current)
                B_current=p_refine.B_new
                KV_current=p_refine.KV_new
                
            """
            If necessary, reduces multiplicity
            """
            if self.mult_xi_new<self.p_new: #p_new is max multipliity of new knots
                t= self.p_new-self.mult_xi_new
                knot_remove=IGA_knot_removal(self.p_new,KV_current,B_current,t)
                B_current=knot_remove.B_new
                KV_current=knot_remove.KV_new
                
            """
            Performs h refinement if neccesary
            """
            if self.num_ele_xi_new>self.num_ele_xi_old: #p_refine will always output C_0 continuous scheme
                n_current=len(KV_current)-(self.p_new+1)
                n_added=len(self.KV_xi_new)-len(KV_current)
                
                h_refine= IGA_h_refine(KV_current,self.KV_xi_new,B_current,n_current,n_added,self.p_new)
                B_current=h_refine.B_new
                KV_current=h_refine.KV_new
            
            
            B_new_x[row,:self.n_new]=B_current[:,0]
            B_new_y[row,:self.n_new]=B_current[:,1]
            B_new_w[row,:self.n_new]=B_current[:,2]
    
        
        B_new_x=B_new_x[:,:self.n_new]
        B_new_y=B_new_y[:,:self.n_new]
        B_new_w=B_new_w[:,:self.n_new]

            
        

        """
        Cycles over eta direction
        """
        
        for col in range(self.n_new):
            B_x_current=B_new_x[:self.m_old,col]
            B_y_current=B_new_y[:self.m_old,col]
            B_w_current=B_new_w[:self.m_old,col]
                            
            B_current=np.vstack((B_x_current,B_y_current,B_w_current)).T
            
            KV_current=self.KV_eta_old
            
            """
            Runs p refine first
            """
            if self.q_new>self.q_old: #p_new must be greater than or equal to p_old
                p_refine=IGA_p_refine(self.q_old,self.q_new,B_current,KV_current)
                B_current=p_refine.B_new
                KV_current=p_refine.KV_new
                
            """
            If necessary, reduces multiplicity
            """                 
            if self.mult_eta_new<self.q_new:
                t= self.q_new-self.mult_eta_new
                knot_remove=IGA_knot_removal(self.q_new,KV_current,B_current,t)
                B_current=knot_remove.B_new
                KV_current=knot_remove.KV_new
            
            """
            Performs h refinement if neccesary
            """                       
            if self.num_ele_eta_new>self.num_ele_eta_old: #p_refine will always output C_0 continuous scheme
                m_current=len(KV_current)-(self.q_new+1)
                m_added=len(self.KV_eta_new)-len(KV_current)
                
                            
                h_refine= IGA_h_refine(KV_current,self.KV_eta_new,B_current,m_current,m_added,self.q_new)
                B_current=h_refine.B_new
                KV_current=h_refine.KV_new
                

            B_new_x[:self.m_new,col]=B_current[:,0]
            B_new_y[:self.m_new,col]=B_current[:,1]
            B_new_w[:self.m_new,col]=B_current[:,2]
        
        
        B_new_x=B_new_x[:self.m_new]
        B_new_y=B_new_y[:self.m_new]
        B_new_w=B_new_w[:self.m_new]
        """
        Reformats B_new into control net form
        """
        B_new_x=B_new_x.flatten()
        B_new_y=B_new_y.flatten()
        B_new_w=B_new_w.flatten()
        
        return np.vstack((B_new_x,B_new_y,B_new_w)).T
    
def refine_multipatch(patch_data_old,patch_data_new,B_old,single_patch=False):
    """
    Refines a multiple patch system. 
    
    INPUTS:
    patch_data_old:  IGA_pre_processing object for coarse mesh
    patch_data_new:  IGA_pre_processing object for refined mesh
    B_old:           Original control net
    """
    order_old=patch_data_old.order
    order_new=patch_data_new.order
    
    multiplicity_old=patch_data_old.mp
    multiplicity_new=patch_data_new.mp
    
    num_el_old=patch_data_old.number_elements
    num_el_new=patch_data_new.number_elements
    
    num_patches=len(order_old)
#    weights=B_old[:,-1]
#    B_old[:,0:-1]=B_old[:,0:-1]*weights[:,None]

    for aa in range(num_patches):
        """
        PLace data into format compatible with surface_refine
        """
        order_data=np.vstack((order_old[aa],order_new[aa])).T
        mult_data=np.vstack((multiplicity_old[aa],multiplicity_new[aa])).T
        num_el_data=np.vstack((num_el_old[aa],num_el_new[aa])).T
        
        """
        Get control point references from pre processing object
        """
        B_old_refs_2D=patch_data_old.patch_global_2D(aa)
        B_old_refs_vec=B_old_refs_2D.flatten()
        
        """
        Find old control points from references
        """
        B_old_current=B_old[B_old_refs_vec]
        
        """
        Run surface refine for current patch
        """        
        B_new_current=Surface_refine(mult_data,num_el_data,order_data,B_old_current).B_new
        
        if single_patch==True:
            return B_new_current
#        if aa==2:
#            return order_data,mult_data,num_el_data,B_old_current,B_new_current
        """
        Start assembling new control net
        """
        if aa==0:
            """
            Initialize B_new for patch 0
            """
            B_new=B_new_current[:]
        else:
            """
            Extra steps for adding control points from other patches
            """
            #Gives size of current B_new
            length_current_net=len(B_new)
            
            #Using "mimic" array of patch global basis numbers, finds indeces of entries greater than current B_new length
            current_refs=np.where(patch_data_new.patch_global_2D(aa)>length_current_net-1)
            
            #Uses current_refs to find local indeces. Local indeces are then used to select desired rows of B_new_current
            local_refs=patch_data_new.local_IEN(aa,1)[current_refs[0],current_refs[1]]
            
            #Uses indeces from prveious step to find the control points that need to be added. This prevents shared control points from being 
            #Added multiple times to new net.
            added_control_points=B_new_current[local_refs]

            #Appends B_new with new control points
            B_new=np.vstack((B_new,added_control_points))
#            if aa==2:
#                return B_new_current,B_new
            
#    weights_new= B_new[:,-1]   
#    B_new[:,0:-1]=B_new[:,0:-1]/weights_new[:,None]    
    return B_new


"""
test
"""
test=IGA_p_refine(1,2,np.array([[0,0,1],[10,0,1]]),np.array([0,0,1,1]))    
