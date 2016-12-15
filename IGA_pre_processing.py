# -*- coding: utf-8 -*-
"""
Created on Sat Jul 04 15:02:52 2015
"""
import numpy as np

from scipy.special import legendre
from scipy.optimize import fsolve

"""
Finds integration points for 2-D
"""
def compute_gauss_points_and_weights(order,dimension):
    """
    INPUTS:
    
    order:     Order of quadrature
    dimension: Option for 1 or 2-D
    """
    
    points,weights=np.polynomial.legendre.leggauss(order)

    if dimension==1:   
        
        return points, weights
    
    else:
        eta_til,xi_til=np.meshgrid(points,points)
        pts_eta_til=eta_til.flatten(); pts_xi_til=xi_til.flatten()
        
        weights_eta,weights_xi=np.meshgrid(weights,weights)
        weights_eta=weights_eta.flatten(); weights_xi=weights_xi.flatten(); 
        
        weights=weights_xi*weights_eta
        
        return pts_eta_til,pts_xi_til,weights
        
    
class patch_data():
    """
    Creates knot vectors and connectivity arrays for given patch data arrays
    """
    def __init__(self,order,number_elements,multiplicity,patch_connect_info,dimension=2):
        """
        INPUTS:
        
        order:                Array containing info on NURBS order
        number_elements:      Array containing info on number of elements
        multiplicity:         Array containing info on knot multiplicity
        patch_connect_info:   Array describing how pacthes are connected to each other
        
        Arrays are structured as follows:
        
        -Each row refers to a patch
        -1st column is data in xi direction, 2nd column is data in eta
        -For single patch arrays, second row contains -1 entries
        -For patch connectivity array, 1st row corresponds to second patch
        """
        self.number_elements=number_elements
        self.order=order
        self.mp=multiplicity
        self.patch_connect_info=patch_connect_info
        self.dimension=dimension
        """
        Save various arrays
        """
        self.create_num_bases()
#        print(self.num_bases)
        self.array_data()
        self.create_Global_IEN()
        
    def knot_vector(self,number_elements,order,multiplicity):
        """
        Computes knot vector assuming open, uniform vector.
        
        INPUTS:
        
        number_elements: Number of elements (number of knot spans)
        order:           Order of NURBS
        multiplicity:    Multiplicity of internal knots
        """
        knot_vector_interior=np.linspace(0,1,number_elements+1)
        knot_vector_interior=np.repeat(knot_vector_interior[1:-1],multiplicity) 
        
        """
        Add zeros to beginning
        """
        knot_vector=np.append(np.zeros(order+1),knot_vector_interior)
        
        """
        Add ones to end
        """
        knot_vector=np.append(knot_vector,np.ones(order+1))
        
        return knot_vector
        
    def create_num_bases(self):
        """
        Creates array for storing the number of bases in xi and eta on each patch
        """
        if self.dimension==1:
             KV_xi=self.knot_vector(self.number_elements,self.order,self.mp)
             self.num_bases=len(KV_xi)-(self.order+1)
             return
        """
        Generates knot vectors for each patch
        """
#        print(self.number_elements)
        KV_xi=lambda patch_num: self.knot_vector(self.number_elements[patch_num,0],self.order[patch_num,0],self.mp[patch_num,0])
        KV_eta=lambda patch_num: self.knot_vector(self.number_elements[patch_num,1],self.order[patch_num,1],self.mp[patch_num,1])
        
        """
        Finds number of bases in knot vectors
        """
        num_basis_xi=lambda patch_num: len(KV_xi(patch_num))-(self.order[patch_num,0]+1)
        num_basis_eta=lambda patch_num: len(KV_eta(patch_num))-(self.order[patch_num,1]+1)
        
        if np.array_equal(self.order[1,:],np.ones(2)*-1)!=1:
            self.num_bases=np.array([ (num_basis_xi(patch_num),num_basis_eta(patch_num)) \
                for patch_num in range(len(self.order))])           
        else:
            self.num_bases=np.vstack((np.array([num_basis_xi(0),num_basis_eta(0)]),np.zeros(2)))
            
                
    def INC(self,patch_num):
        """
        Finds INC based off of basis order and knot vector. INC relates local basis numbers
        to eachother
        """
        if self.dimension==1:
              num_basis_xi=self.num_bases
              i_arr = np.arange(num_basis_xi, dtype=np.int)
              return (np.array([ (i) for  i in i_arr ], dtype=np.int).transpose())
            
        elif self.dimension==2:   
            num_basis_xi=self.num_bases[patch_num,0]
            num_basis_eta=self.num_bases[patch_num,1]

            #Index arrays in each basis direction
            i_arr = np.arange(num_basis_xi, dtype=np.int)
            j_arr = np.arange(num_basis_eta, dtype=np.int)
    
            #Construct the coordinate array
            return (np.array([ (i, j) for j in j_arr for i in i_arr ], dtype=np.int).transpose())
    
    
    def local_IEN(self,patch_num,patch_mimic=False):
        """
        Finds IEN over patch.  Local IEN describes distribution of local basis numbers into
        patch elements
        
        INPUTS:
        patch_num:   Gives number of row to reference in data arrays
        patch_mimic: Option for outputting local_IEN into form that resembles number of bases in each direction
        """     
        """
        1D case
        """
       
        if self.dimension==1:
            num_bases=self.num_bases
            num_element_bases=self.order+1
            num_local_bases=num_bases
            basis_array=np.linspace(1,num_local_bases,num_local_bases)

            if patch_mimic:
                return basis_array.astype(int)-1 #MAY NEED TO REVERSE
            else:  
                

                """
                Total number of elements in patch
                """
                num_elements=self.number_elements
                
                """
                Initializing IEN array
                """
                if num_elements==1:
                    patch_local_IEN=np.zeros(num_element_bases)
                   
                else:
                    patch_local_IEN=np.zeros((num_elements,num_element_bases))
                
                """
                counter for IEN row
                """
                IEN_row=0
                """
                for loops for finding entries for each row of IEN
                """
                for col in range(self.number_elements):
                                            
                    """
                    Bounds for rows and columns in basis_array for current element
                    """
                    lowest_col_in_ele=col*self.mp
                    highest_col_in_ele=col*self.mp+self.order+1 #formatted to be used as index

                    
                    """
                    Gathers entries for current element in local IEN
                    """
                    row_local_IEN=basis_array[lowest_col_in_ele:highest_col_in_ele]
                    
                    if num_elements==1:
                        patch_local_IEN=row_local_IEN[::-1]
                    else:
                        patch_local_IEN[IEN_row,:]=row_local_IEN[::-1]
                    
                    
                    """
                    Counter for going to next row in IEN
                    """
                    IEN_row+=1

                
                """
                Ensuring that entry is a 2D array by using a dummy row for consistency
                """
                if len(patch_local_IEN.shape)!=2 :
                     patch_local_IEN=np.vstack((patch_local_IEN,np.zeros(len(patch_local_IEN))))
                  
                return patch_local_IEN.astype(int)-1
        """
        2D case
        """
        """
        Number of bases in principle directions along patch
        """
        
        num_basis_xi=self.num_bases[patch_num,0]
        num_basis_eta=self.num_bases[patch_num,1]
                
        """
        Total number of bases functions over patch
        """
        num_local_bases=num_basis_xi*num_basis_eta
        
        """
        Number of supporting bases over an element
        """
        dimensions=self.order[patch_num,:]+1 #Number of bases in xi and eta direction with support on each element
        num_element_bases=dimensions.prod() 
        
        """
        Creating 2d array in "shape" of elements in patch that contains basis function numbers
        """
        basis_array=np.linspace(1,num_local_bases,num_local_bases)
        basis_array=basis_array.reshape(num_basis_eta,num_basis_xi)
        
        if patch_mimic:
           
            return basis_array.astype(int)-1 #MAY NEED TO REVERSE
            
        else:     
            
            """
            Total number of elements in patch
            """
            num_elements=self.number_elements[patch_num,:].prod()
            
            """
            Initializing IEN array
            """
            if num_elements==1:
                patch_local_IEN=np.zeros(num_element_bases)
            else:
                patch_local_IEN=np.zeros((num_elements,num_element_bases))
            
            """
            counter for IEN row
            """
            IEN_row=0
            
            """
            for loops for finding entries for each row of IEN
            """
            for row in range(self.number_elements[patch_num,1]):
                for col in range(self.number_elements[patch_num,0]):
                    
                    #ASK about line 294 in IGA file to shorten this
                    
                    """
                    Bounds for rows and columns in basis_array for current element
                    """
                    lowest_row_in_ele=row*self.mp[patch_num,1]
                    highest_row_in_ele=row*self.mp[patch_num,1]+self.order[patch_num,1]+1 #formatted to be used as index
                    lowest_col_in_ele=col*self.mp[patch_num,0]
                    highest_col_in_ele=col*self.mp[patch_num,0]+self.order[patch_num,0]+1 #formatted to be used as index
                    
                    """
                    Gathers entries for current element in local IEN
                    """
                    row_local_IEN=basis_array[lowest_row_in_ele:highest_row_in_ele,lowest_col_in_ele:highest_col_in_ele]
                    
                    if num_elements==1:
                        patch_local_IEN=row_local_IEN.flatten()[::-1]
                    else:
                        patch_local_IEN[IEN_row,:]=row_local_IEN.flatten()[::-1]
                    
                    
                    """
                    Counter for going to next row in IEN
                    """
                    IEN_row+=1
            
            """
            Ensuring that entry is a 2D array by using a dummy row for consistency
            """
            if len(patch_local_IEN.shape)!=2 :
                 patch_local_IEN=np.vstack((patch_local_IEN,np.zeros(len(patch_local_IEN))))
                 
            return patch_local_IEN.astype(int)-1
    
    def find_Global_IEN_indeces(self,shared_patch_num,boundary_dir,boundary_end):
        """
        Uses local_IEN and 2D array to find indeces of bases with support along booundary in Global_IEN
        
        INPUTS:
        shared_patch_num:   number of patch number that shares boundary with current patch in create_Global_IEN
        boundary_dir:       direction of shared boundary
        boundary_end:       0 or 1 with respect to shared patch
        """
           
        """
        2D case
        """
        patch_local_IEN=self.local_IEN(shared_patch_num)
        patch_local_IEN_2D=self.local_IEN(shared_patch_num,1)
        
        """
        Number of bases in principle directions along sharing patch
        """
        num_basis_xi=self.num_bases[shared_patch_num,0]
        num_basis_eta=self.num_bases[shared_patch_num,1]
        
        """
        Patches are 1-1 along boundaries
        """
        if boundary_dir==0 and boundary_end==0:
            shared_bases=patch_local_IEN_2D[0,:]
            
        elif boundary_dir==0 and boundary_end==1:
            shared_bases=patch_local_IEN_2D[num_basis_eta-1,:]
            
        elif boundary_dir==1 and boundary_end==0:
            shared_bases=patch_local_IEN_2D[:,0]
        
        elif boundary_dir==1 and boundary_end==1:
            shared_bases=patch_local_IEN_2D[:,num_basis_xi-1] 
     
        """
        Outputs indeces of shared bases
        """
        for i in np.arange(len(shared_bases)):
            if i==0:
                row_index=np.where(patch_local_IEN==shared_bases[i])[0][0] #Specifes only first instance to avoid repeated indeces
                col_index=np.where(patch_local_IEN==shared_bases[i])[1][0]
                Global_IEN_indeces=np.vstack((row_index,col_index))
            else:
                row_index=np.where(patch_local_IEN==shared_bases[i])[0][0]
                col_index=np.where(patch_local_IEN==shared_bases[i])[1][0]
                Global_IEN_indeces_i=np.vstack((row_index,col_index))
                Global_IEN_indeces=np.hstack((Global_IEN_indeces,Global_IEN_indeces_i))
        
        #row index on top, column index on bottom
        return Global_IEN_indeces.astype(int)
        
#        np.where(patch_local_IEN==i for i in shared_bases)
        
    def create_Global_IEN(self):
        """
        Creates Global IEN array and secondary array describing which rows belong to which patch
        
        Structure of the global IEN array:
        -Blocks of rows, referenced by the second output, refer to the global basis numbers contained within each patch
        -Each row of each block refers to an element in that patch
        -Blocks containing info for only one element will have a second row containing all -1 entries
        -Since all patches may not have the same number of bases in the principal directions, -1's are added to the rows of the IEN
         where necessary in order to get a valid matrix
        
        Structure of reference array ("length" array):
        -Row i corresponds to patch i
        -1st column is starting row, 2nd column is ending row corresponding to patch in global IEN
        -If reference is to single element patch, both emtries in row will reference non dummy row in global IEN
        """
        
        """
        1D case
        """
        if self.dimension==1:
              self.Global_IEN=self.local_IEN(0)
              self.Global_IEN_length=np.array([[0,0],[-1,-1]])
              return
            
        """
        2D case
        """
        patch_connect_info=self.patch_connect_info
        
        """
        Cycles through patches during connectivity assembly
        """
        for patch_num in range(len(self.num_bases)): #len(self.order)
            """
            Initializes array
            """
            if patch_num==0:
                """
                Handles condition if only one patch is present
                """
                if np.array_equal(self.order[1],np.ones(2)*-1)==1: #ends Global IEN in case of single patch
                    self.Global_IEN=self.local_IEN_array[0]
                    self.Global_IEN_length=np.array([[0,0],[-1,-1]])
                    return
                    
                """
                Number of bases in principle directions along patch
                """
                num_basis_xi=self.num_bases[patch_num,0]
                num_basis_eta=self.num_bases[patch_num,1]
                
                """
                Starts Global_IEN definition for multiple patch systems
                """
                Global_IEN_row=self.local_IEN_array[0]+1
                
                """
                Last row in Global_IEN relevant to current patch
                """
                first_index_length=0
                end_index_length=len(Global_IEN_row)-1 #Indexed such that 1 doesn't have to be added in row reference

                """
                Gives rows in Global_IEN that correspond to each patch
                """
                self.Global_IEN_length=np.array([first_index_length,end_index_length])
                self.Global_IEN=Global_IEN_row
               
                """
                starting index number for next patch (patch 2)
                """
                start_index_IEN=np.max(Global_IEN_row)+1
                start_index_length=len(self.Global_IEN)
            
            else:
                """
                Assembles matrices for all patches after patch 1
                """
                     
                """
                Number of bases in principle directions along patch
                """
                num_basis_xi=self.num_bases[patch_num,0]
                num_basis_eta=self.num_bases[patch_num,1]
                total_bases_current_patch=num_basis_xi*num_basis_eta
                
                """
                Initializes shape of current patch and gives shape of previous patch
                """
                
                shape_current_patch=np.zeros((num_basis_eta,num_basis_xi))
#                print(shape_current_patch,self.num_bases)
                
                    
                """
                Loops though patches that are shared with current patch
                """
                for i in range(len(patch_connect_info[patch_num-1,:,0])): #patch_connect_info starts info at patch 2
                    
                    """
                    Flag for elements with less than the max amount of patch data
                    
                    ex: If patch 2 shares a boundary with patch 1 (1 shared boundary) , and patch 4 shares a boundary with patch 
                        2 and 3 (2 shared boundaries)
                    """
                    if patch_connect_info[patch_num-1,i,0]==-1:
                        continue
                    
                    """
                    Patch along which current patch shares boundary
                    """
                    shared_patch_num=patch_connect_info[patch_num-1,i,0]
                    patch_boundary_dir=patch_connect_info[patch_num-1,i,1]
                    patch_boundary_end=patch_connect_info[patch_num-1,i,2]    
                    
                    """
                    Picks  rows out of Global_IEN corresponding to sharing patch. Rows for the shared patches
                    should already have been created in previous loop iterations.
                    """
                    if len(self.Global_IEN_length.shape)!=2:
                        start_row=self.Global_IEN_length[0]
                        end_row=self.Global_IEN_length[1]
                        shared_patch_rows=self.Global_IEN[start_row:end_row+1,:]
                    
                    else:
                        start_row=self.Global_IEN_length[shared_patch_num,0]
                        end_row=self.Global_IEN_length[shared_patch_num,1]
                        shared_patch_rows=self.Global_IEN[start_row:end_row+1,:]
                    
                    """
                    Finds indeces of sharing patch rows which give bases shared along boundary
                    """
                    indeces=self.find_Global_IEN_indeces(shared_patch_num,patch_boundary_dir,patch_boundary_end)   
                    x_ind=indeces[0,:]
                    y_ind=indeces[1,:]
                    
                    """
                    Applies indeces to get common Global bases function numbers along boundary
                    """
                    shared_bases=shared_patch_rows[x_ind,y_ind]
               
                                     
                    """
                    Fills in current patch 2D array with common bases numbers
                    """
                    if patch_boundary_dir==0 and patch_boundary_end==1:
                        shape_current_patch[0,:]=shared_bases
        
                    elif patch_boundary_dir==0 and patch_boundary_end==0:
                        shape_current_patch[num_basis_eta-1,:]=shared_bases
                   
                    elif patch_boundary_dir==1 and patch_boundary_end==1:
#                        print(shape_current_patch,shared_bases)
                        shape_current_patch[:,0]=shared_bases
                    
                    elif patch_boundary_dir==1 and patch_boundary_end==0:
                        shape_current_patch[:,num_basis_xi-1]=shared_bases
                    
                  
                """
                Finds indeces of remianing zero entries that need to be filled
                """
                remaining_entries_indeces=np.where(shape_current_patch==0)   
                remaining_entries_indeces_xi=remaining_entries_indeces[0]
                remaining_entries_indeces_eta=remaining_entries_indeces[1]
                    
                """
                Total number of remianing zero entries that need to be replaced
                """
                num_remaining=len(remaining_entries_indeces_xi)
                
                """
                Numbers to be substituted into current patch 2D array
                """
                new_global_nums=np.linspace(start_index_IEN,start_index_IEN+num_remaining-1,num_remaining)
                                
                """
                replacing zeros
                """
                shape_current_patch[remaining_entries_indeces]=new_global_nums
                
                """
                Number of elements in current patch
                """
                num_elements_current_patch_xi=self.number_elements[patch_num,0]
                num_elements_current_patch_eta=self.number_elements[patch_num,1]
                total_elements_current_patch=num_elements_current_patch_xi*num_elements_current_patch_eta
                              
                """
                Counter for IEN row
                """
                IEN_row=0
                
                """
                Number of supporting bases over an element
                """
                dimensions=self.order[patch_num,:]+1 #Number of bases in xi and eta direction with support on each element
                num_element_bases=dimensions.prod() 
                
                """
                Initializing IEN array
                """
                if total_elements_current_patch==1:
                    patch_Global_IEN=np.zeros(total_elements_current_patch)
                else:
                    patch_Global_IEN=np.zeros((total_elements_current_patch,num_element_bases))
    
                
                """
                Applies procedure similar to that in local_IEN for changing format from 2D array
                """
                for row in range(num_elements_current_patch_eta):
                    for col in range(num_elements_current_patch_xi):
                        
                        #ASK about line 294 in IGA file to shorten this
                        
                        """
                        Bounds for rows and columns in basis_array for current element
                        """
                        lowest_row_in_ele=row*self.mp[patch_num,1]
                        highest_row_in_ele=row*self.mp[patch_num,1]+self.order[patch_num,1]+1 #formatted to be used as index
                        lowest_col_in_ele=col*self.mp[patch_num,0]
                        highest_col_in_ele=col*self.mp[patch_num,0]+self.order[patch_num,0]+1 #formatted to be used as index
                        
                        """
                        gathers entries for current element in Global IEN
                        """
                        row_Global_IEN=shape_current_patch[lowest_row_in_ele:highest_row_in_ele,lowest_col_in_ele:highest_col_in_ele]
                        
                        if total_elements_current_patch==1:
                             patch_Global_IEN=row_Global_IEN.flatten()[::-1]
                        else:
                             patch_Global_IEN[IEN_row,:]=row_Global_IEN.flatten()[::-1]
                 
                        """
                        Counter for going to next row in IEN
                        """
                        IEN_row+=1
                
                
                
                """
                Ensuring that entry is a 2D array by using a dummy row for consistency
                """
                if len(patch_Global_IEN.shape)!=2 :
                     patch_Global_IEN=np.vstack((patch_Global_IEN,np.zeros(len(patch_Global_IEN))))
                
                """
                Gives rows of Global_IEN that correspond to current patch
                """
                if total_elements_current_patch==1:
                    current_length_array=np.array([start_index_length,start_index_length+1])
                else:
                    current_length_array=np.array([start_index_length,start_index_length+total_elements_current_patch-1])

                """
                Modifies Global_IEN by adding zeros to ends of patch_data with less entries than max
                """
                if self.Global_IEN.shape[1]<patch_Global_IEN.shape[1]:
                    num_zeros_adding_on=patch_Global_IEN.shape[1]-self.Global_IEN.shape[1]
                    num_rows=len(self.Global_IEN)
                    zeros_array=np.zeros((num_rows,num_zeros_adding_on))
                    self.Global_IEN=np.hstack((self.Global_IEN,zeros_array))
                
                """
                If current patch rows have less entries in column than current Global_IEN, add appropriate number of zeros
                """
                if self.Global_IEN.shape[1]>patch_Global_IEN.shape[1]:
                    num_zeros_adding_on=-patch_Global_IEN.shape[1]+self.Global_IEN.shape[1]
                    num_rows=len(patch_Global_IEN)
                    zeros_array=np.zeros((num_rows,num_zeros_adding_on))
                    patch_Global_IEN=np.hstack((patch_Global_IEN,zeros_array))
                    
                self.Global_IEN=np.vstack((self.Global_IEN,patch_Global_IEN)).astype(int)
                self.Global_IEN_length=np.vstack([self.Global_IEN_length,current_length_array])
                
                """
                Changing numbers from which next iteration will start indexing
                """
                start_index_IEN=np.max(patch_Global_IEN)+1
                start_index_length=len(self.Global_IEN)
        """
        Subtract 1 to remain consistent with python indeces
        """        
        self.Global_IEN=self.Global_IEN-1
                                        
    def patch_global_2D(self,patch_num):
        """
        Takes info from global IEN for speciifed patch and rearranges it into "mimic" 2D array
        
        INPUTS:
        patch_num:  Patch number
        """
        
        """
        Checks for simple case of single element patch
        """
        """
        1D case
        """
        if self.dimension==1:
            return self.local_IEN(0,1)

        """
        2D case
        """        
        if np.array_equal(self.order[1],np.ones(2)*-1)==1: #ends Global IEN in case of single patch
            
            return self.local_IEN(0,1)
            
        patch_local_IEN=self.local_IEN(patch_num,0)
        if np.prod(self.number_elements[patch_num])==1:
            patch_local_IEN=patch_local_IEN[0]
            
        patch_local_IEN_u=np.unique(patch_local_IEN,return_index=True)
             
        patch_global_info=self.Global_IEN[self.Global_IEN_length[patch_num,0]:self.Global_IEN_length[patch_num,1]+1,:]
        
        """
        Removes unecessary 0's along axis 0
        """
        patch_global_info=patch_global_info[:,0:np.prod(self.order[patch_num]+1)]
        
        """
        Removes unecessary 0's in axis 1
        """
        if np.prod(self.number_elements[patch_num])==1:
            patch_global_info=patch_global_info[0]
                  
        return (patch_global_info.flatten()[patch_local_IEN_u[1]]).reshape(self.num_bases[patch_num][::-1])

    def array_data(self):
        """
        Creates arrays of knot vectors, IEN, etc to reduce number of calculations in other files
        """
        self.INC_array=[]
        self.local_IEN_array=[]
        self.KV_xi_array=[]
        self.KV_eta_array=[]
        
        if self.dimension==1:
            
            self.INC_array.append(self.INC(0))
            self.local_IEN_array.append(self.local_IEN(0))
            self.KV_xi_array.append(self.knot_vector(self.number_elements,self.order,self.mp))
        
        elif self.dimension==2:
        
            for pnum in np.arange(len(self.num_bases)):
                
                self.INC_array.append(self.INC(pnum))
                self.local_IEN_array.append(self.local_IEN(pnum))
                self.KV_xi_array.append(self.knot_vector(self.number_elements[pnum,0],self.order[pnum,0],self.mp[pnum,0]))
                self.KV_eta_array.append(self.knot_vector(self.number_elements[pnum,1],self.order[pnum,1],self.mp[pnum,1]))

"""
test data
"""
test= patch_data(2,3,1,0,1)