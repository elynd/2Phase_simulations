"""
Finite Element Methods
Coding Project 2
3/23/15
@author: Eric Lynd
"""
import numpy as np
import netCDF4 #allows import of Sandia's Exodus database formatted files
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
#import scipy.optimize.curve_fit
from scipy.interpolate import griddata 
from scipy.interpolate import interp2d 
import matplotlib
pgf_with_custom_preamble = {
    "font.family": "serif",   # use serif/main font for text elements
    "text.usetex": True,    # use inline math for ticks
    "pgf.rcfonts": False,   # don't setup fonts from rc parameters
}
matplotlib.rcParams.update(pgf_with_custom_preamble)
import matplotlib.pyplot as plt
import matplotlib.path as mplPath
from sympy.solvers import solve
from sympy import symbols
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

class SFEM_2D():
    
    def __init__(self, degree,file_path,k_term,f_term,essential_vals,source_vals): #bc_array
        """
        Takes in mesh data from CUBIT file and used linear FEM to solve. Applies constant boundary conditions
        across top and bottom of ex. problem domain
        """
        self.degree=degree
        
        data=netCDF4.Dataset(file_path)
        self.data=data
        self.nelem=len(data.variables['connect1'][:,:])
        self.nnodes=len(data.variables['coordx'][:])
        self.x=data.variables['coordx'][:]*100.
        self.y=data.variables['coordy'][:]*100.
        
#        self.x=np.round(self.x/5)*5
#        self.y=np.round(self.y/5)*5
        
#        self.x=self.x-self.x%5
#        self.y=self.y-self.y%5
##        else:
#            self.x=np.round(self.x,1)
#            self.y=np.round(self.y,1)

        self.con_array=data.variables['connect1'][:,:]-1 #accoutning for python indecies
        
        self.k_term=k_term
        self.f_term=f_term
        
        self.num_ess_nodes=len(data.variables['node_ns4'][:])+len(data.variables['node_ns2'][:])+len(data.variables['node_ns3'][:])+len(data.variables['node_ns5'][:])
        self.essential_nodes=np.hstack((data.variables['node_ns4'][:],data.variables['node_ns2'][:],data.variables['node_ns3'][:],data.variables['node_ns5'][:]))-1
        
        
        
        
        
        self.essential_nodes_corners=data.variables['node_ns6'][:]-1        
        
        """
        Data for nodes at corners and midpoints
        """        
        con=np.vstack((self.x,self.y))
        con=con.T
        index1=np.where(np.all(con==np.array([-400,0]),axis=1))[0]
        index2=np.where(np.all(con==np.array([400,0]),axis=1))[0]
        index3=np.where(np.all(con==np.array([0,400]),axis=1))[0]
        index4=np.where(np.all(con==np.array([0,-400]),axis=1))[0]
        
#        print(index1,index2)
#        self.essential_nodes_corners_and_interior=data.variables['node_ns7'][:]-1   
        self.essential_nodes_corners_and_interior=np.hstack((self.essential_nodes_corners,index1,index2,index3,index4))
        
        self.essential_nodes_top_bottom=np.hstack((self.essential_nodes_corners,index3,index4))
        
        self.essential_nodes_left_right=np.hstack((self.essential_nodes_corners,index1,index2))

#        ,index3,index4
        self.essential_nodes_interior=np.hstack((index1,index2,index3,index4))
        
        self.essential_nodes_corners=np.hstack((self.essential_nodes_corners))
        
#        self.bc_array=bc_array
        self.essential_node_vals=np.hstack((np.ones(len(data.variables['node_ns4'][:]))*essential_vals[0], \
                    np.ones(len(data.variables['node_ns2'][:]))*essential_vals[1]))
#                    ,np.ones(len(data.variables['node_ns3'][:]))*essential_vals[2] \
#                    ,np.ones(len(data.variables['node_ns5'][:]))*essential_vals[3]))
                    
                    
        
    
        self.nodeload=data.variables['node_ns1'][:]-1
        self.fhat=np.ones(len(data.variables['node_ns1'][:]))*source_vals
        
        side_to_local_index = np.array([[0,1],[1,2],[2,3],[0,3]], dtype=np.int)

        """
        Applying Nuemann conditions at corners
        """
        self.elements_in_sideset = data.variables['elem_ss1'][:] - 1        
        global_nodes_in_sideset = self.con_array[self.elements_in_sideset]
        self.sides_in_sideset=data.variables['side_ss1'][:]-1
        self.local_nodes_in_side_set = side_to_local_index[self.sides_in_sideset[:]]
        self.global_nodes_in_sideset=global_nodes_in_sideset[np.arange(self.local_nodes_in_side_set.shape[0])[:,None], self.local_nodes_in_side_set]
        
        """
        Applying Nuemann condition along edges
        """
        ###Bottom edge
        self.nodes_bottom=data.variables['node_ns2'][:]
        self.elements_bottom = data.variables['elem_ss2'][:] - 1
        global_nodes_bottom = self.con_array[self.elements_bottom] 
        self.sides_bottom=data.variables['side_ss2'][:]-1
        self.local_nodes_bottom = side_to_local_index[self.sides_bottom[:]]
        self.global_nodes_bottom=global_nodes_bottom[np.arange(self.local_nodes_bottom.shape[0])[:,None], self.local_nodes_bottom]
        
        ###Left edge
        self.nodes_left=data.variables['node_ns3'][:]
        self.elements_left= data.variables['elem_ss3'][:] - 1
        global_nodes_left= self.con_array[self.elements_left] 
        self.sides_left=data.variables['side_ss3'][:]-1
        self.local_nodes_left = side_to_local_index[self.sides_left[:]]
        self.global_nodes_left=global_nodes_left[np.arange(self.local_nodes_left.shape[0])[:,None], self.local_nodes_left]
        
        ###Top edge
        self.nodes_top=data.variables['node_ns5'][:]
        self.elements_top= data.variables['elem_ss5'][:] - 1
        global_nodes_top= self.con_array[self.elements_top] 
        self.sides_top=data.variables['side_ss5'][:]-1
        self.local_nodes_top = side_to_local_index[self.sides_top[:]]
        self.global_nodes_top=global_nodes_top[np.arange(self.local_nodes_top.shape[0])[:,None], self.local_nodes_top]
        
        ###Right edge
        self.nodes_right=data.variables['node_ns4'][:]
        self.elements_right= data.variables['elem_ss4'][:] - 1
        global_nodes_right= self.con_array[self.elements_right] 
        self.sides_right=data.variables['side_ss4'][:]-1
        self.local_nodes_right = side_to_local_index[self.sides_right[:]]
        self.global_nodes_right=global_nodes_right[np.arange(self.local_nodes_right.shape[0])[:,None], self.local_nodes_right]
        
        
#        self.source_strength=source_vals/(len(self.elements_right)*2)
        self.source_strength=source_vals

        
        #Runs function to generate approximation at nodes
        self.nodal_values()
        
    def N(self,xi,eta,degree):
        """
        Finds shape functions for quad element. Currently only handles linears. 
        Taken from http://nbviewer.ipython.org/github/johnfoster-pge-utexas/PGE383-AdvGeomechanics/blob/master/files/assignment5_solution.ipynb
        """
#        xi_locs=np.linspace(-1,1,self.degree+1)
#        eta_locs=np.linspace(-1,1,self.degree+1)
        
        if degree==1:
            
            return np.array([1 / 4. - eta / 4. - xi / 4. + (eta * xi) / 4.,
                    1 / 4. - eta / 4. + xi / 4. - (eta * xi) / 4., 
                    1 / 4. + eta / 4. + xi / 4. + (eta * xi) / 4.,
                    1 / 4. + eta / 4. - xi / 4. - (eta * xi) / 4.])
                
        elif degree==2:
            
            return np.array([[(eta*xi)/4. - (eta**2*xi)/4. - (eta*xi**2)/4. + (eta**2*xi**2)/4.,
                -(eta*xi)/4. + (eta**2*xi)/4. - (eta*xi**2)/4. + (eta**2*xi**2)/4.,
                (eta*xi)/4. + (eta**2*xi)/4. + (eta*xi**2)/4. + (eta**2*xi**2)/4.,
                -(eta*xi)/4. - (eta**2*xi)/4. + (eta*xi**2)/4. + (eta**2*xi**2)/4.,
                -eta/2. + eta**2/2. + (eta*xi**2)/2. - (eta**2*xi**2)/2.,
                xi/2. - (eta**2*xi)/2. + xi**2/2. - (eta**2*xi**2)/2.,
                eta/2. + eta**2/2. - (eta*xi**2)/2. - (eta**2*xi**2)/2.,
                -xi/2. + (eta**2*xi)/2. + xi**2/2. - (eta**2*xi**2)/2.,
                1 - eta**2 - xi**2 + eta**2*xi**2]])
                    
    def dNdxi(self,xi,eta,degree):
        """Compute shape function derivatives with respect to xi"""
        
        if degree==1:
            
            return np.array([-1 / 4. + eta / 4., 
                     1 / 4. - eta / 4., 
                     1 / 4. + eta / 4., 
                    -1 / 4. - eta / 4.])
                    
        if degree==2:
            
            return np.array([eta/4. - eta**2/4. - (eta*xi)/2. + (eta**2*xi)/2.,
                -eta/4. + eta**2/4. - (eta*xi)/2. + (eta**2*xi)/2.,
                eta/4. + eta**2/4. + (eta*xi)/2. + (eta**2*xi)/2.,
                -eta/4. - eta**2/4. + (eta*xi)/2. + (eta**2*xi)/2.,eta*xi - eta**2*xi,
                0.5 - eta**2/2. + xi - eta**2*xi,-(eta*xi) - eta**2*xi,
                -0.5 + eta**2/2. + xi - eta**2*xi,-2*xi + 2*eta**2*xi])
            
                
                
    def dNdeta(self,xi,eta,degree):
        """Compute shape function derivatives with respect to eta"""
        
        if degree==1:
            
            return np.array([-1 / 4. + xi / 4.,
                    -1 / 4. - xi / 4.,
                     1 / 4. + xi / 4.,
                     1 / 4. - xi / 4.])  
                     
        elif degree==2:
             
            return np.array([xi/4. - (eta*xi)/2. - xi**2/4. + (eta*xi**2)/2.,
                -xi/4. + (eta*xi)/2. - xi**2/4. + (eta*xi**2)/2.,
                xi/4. + (eta*xi)/2. + xi**2/4. + (eta*xi**2)/2.,
                -xi/4. - (eta*xi)/2. + xi**2/4. + (eta*xi**2)/2.,
                -0.5 + eta + xi**2/2. - eta*xi**2,-(eta*xi) - eta*xi**2,
                0.5 + eta - xi**2/2. - eta*xi**2,eta*xi - eta*xi**2,-2*eta + 2*eta*xi**2])
                
    def compute_gauss_points_and_weights(self,order,dimension):
        """
        Finds integration points
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
            
    def jacobian(self,xi,eta,element,inverse=False):
        """
        Finds J or J^-1 for a quad element with linear bases. 

        """
        con=self.con_array[element,:] #row of connect. array corresponding to input element number
        x=self.x[con] 
        y=self.y[con]
        
        J11=np.dot(self.dNdxi(xi,eta,self.degree),x)
        J12=np.dot(self.dNdeta(xi,eta,self.degree),x)
        J21=np.dot(self.dNdxi(xi,eta,self.degree),y)
        J22=np.dot(self.dNdeta(xi,eta,self.degree),y)
        
        J=np.array([[J11,J12],[J21,J22]])
        
        if inverse:
            return np.linalg.inv(J)
        else:
            return J
     
    def assemble(self,test=False):
        """
        Builds stiffness matrix and force vector. 
        """
        
        #Initializing matrices
        self.K=np.zeros((self.nnodes,self.nnodes))
        self.F=np.zeros(self.nnodes)
        
        k=self.k_term
        f=self.f_term
        
        pts_xi,pts_eta,weights=self.compute_gauss_points_and_weights(2,2)
        #make k term and f term more genral
        
        for i in range(self.nelem):
            #Connectivity array for element
            con=self.con_array[i,:]
            idx_grid = np.ix_(con, con)
#            grid1,grid2=np.meshgrid(con,con)
            
            
            for j in range(len(weights)):
                    
                #Need to solve jacobian and shape function for each integration point
                Jinv=self.jacobian(pts_xi[j],pts_eta[j],i,True)
                detJ=np.linalg.det(self.jacobian(pts_xi[j],pts_eta[j],i))
                if j==0:
                    Jinv_mat=Jinv
                    detJ_vec=detJ
                else:
                    Jinv_mat=np.dstack((Jinv_mat,Jinv))
                    detJ_vec=np.append(detJ_vec,detJ)
                    
            #Can probably make code  faster by placing this outside inner loop. Otherwise, will reevaluate functions for each elements
            dNdxi=self.dNdxi(pts_xi,pts_eta,self.degree)
            dNdeta=self.dNdeta(pts_xi,pts_eta,self.degree)
            N=self.N(pts_xi,pts_eta,self.degree)
            
            x_pt=np.einsum('ji,j->i',N,self.x[con])
            y_pt=np.einsum('ji,j->i',N,self.y[con])
            
            #output map as array???
            k_vec=k(x_pt,y_pt)
            f_vec=f(x_pt,y_pt)
            
            #combines k or f term, weight, and detJ for each integration point
            lump_vec_k=k_vec*weights*detJ_vec
            lump_vec_f=f_vec*weights*detJ_vec
            
            #Use chain rule to get derivatives in terms of dx and dy
         
#            dNdx=np.einsum('ji,i->ji',dNdxi,Jinv_mat[0,0,:])+np.einsum('ji,i->ji',dNdeta,Jinv_mat[1,0,:])
#            dNdx_grid=np.einsum('ji,ki->jki',dNdx,dNdx)
#            
#            dNdy=np.einsum('ji,i->ji',dNdxi,Jinv_mat[0,1,:])+np.einsum('ji,i->ji',dNdeta,Jinv_mat[1,1,:])
#            dNdy_grid=np.einsum('ji,ki->jki',dNdy,dNdy)
            
            dNdx=np.einsum('ij,j->ij',dNdxi,Jinv_mat[0,0,:])+np.einsum('ij,j->ij',dNdeta,Jinv_mat[1,0,:])
            dNdx_grid=np.einsum('ji,ki->jki',dNdx,dNdx)
            
            dNdy=np.einsum('ij,j->ij',dNdxi,Jinv_mat[0,1,:])+np.einsum('ij,j->ij',dNdeta,Jinv_mat[1,1,:])
            dNdy_grid=np.einsum('ji,ki->jki',dNdy,dNdy)
            
            self.K[idx_grid]+=np.einsum('ijk,k',(dNdx_grid+dNdy_grid),lump_vec_k)
#            self.K[idx_grid]+=np.sum(dNdx_grid+dNdy_grid,axis=2)
            self.F[con]+=np.einsum('ij,j',N,lump_vec_f)
            if test==True:
                return Jinv_mat[:,:,0],detJ_vec[0],np.sum(dNdy_grid+dNdx_grid,axis=2),np.einsum('ijk,k',(dNdy_grid+dNdx_grid),lump_vec_k)
    def edge_integral(self,test=False):
        """
        Nuemann condition for flux at edges
        """
        
        
        self.assemble()
        #Integration points along each element side
        pts_xi,weights=self.compute_gauss_points_and_weights(3,1)   
        
        #Generates basis function values for each integration point
        filler=np.ones(len(pts_xi))*-1 #arbitarry vector used to generate characteristic values for each basis function at int. points
        
        boundary_node_1_vals=self.N(pts_xi,filler,self.degree)[0,:] #values for first node in element side
        boundary_node_2_vals=self.N(pts_xi,filler,self.degree)[1,:] #"" "" second node
        
        """
        Left
        """
        
        #Determinates of mappings between segment and parent element for each element
        x_vals=self.x[self.global_nodes_left]
        y_vals=self.y[self.global_nodes_left]
        
        detJ_left=np.sqrt((y_vals[:,1]-y_vals[:,0])**2+(x_vals[:,1]-x_vals[:,0])**2)/2
    
        for side_num in range(int(len(self.elements_left))):
            side_nodes=self.global_nodes_left[side_num]
            side_detJ=detJ_left[side_num] #constant on each side
            integral_1=np.dot(boundary_node_1_vals,weights)*side_detJ*self.source_strength
            integral_2=np.dot(boundary_node_2_vals,weights)*side_detJ*self.source_strength
            self.F[side_nodes[0]]+=integral_1
            self.F[side_nodes[1]]+=integral_2
         
        """
        Top
        """
          
        #Determinates of mappings between segment and parent element for each element
        x_vals=self.x[self.global_nodes_top]
        y_vals=self.y[self.global_nodes_top]
        
        detJ_top=np.sqrt((y_vals[:,1]-y_vals[:,0])**2+(x_vals[:,1]-x_vals[:,0])**2)/2

        for side_num in range(int(len(self.elements_top))):
            side_nodes=self.global_nodes_top[side_num]
            side_detJ=detJ_top[side_num] #constant on each side
            integral_1=np.dot(boundary_node_1_vals,weights)*side_detJ*self.source_strength
            integral_2=np.dot(boundary_node_2_vals,weights)*side_detJ*self.source_strength
            self.F[side_nodes[0]]+=integral_1
            self.F[side_nodes[1]]+=integral_2 
         
        """
        Bottom
        """
          
        #Determinates of mappings between segment and parent element for each element
        x_vals=self.x[self.global_nodes_bottom]
        y_vals=self.y[self.global_nodes_bottom]
        
        detJ_bottom=np.sqrt((y_vals[:,1]-y_vals[:,0])**2+(x_vals[:,1]-x_vals[:,0])**2)/2

        for side_num in range(int(len(self.elements_bottom))):
            side_nodes=self.global_nodes_bottom[side_num]
            side_detJ=detJ_bottom[side_num] #constant on each side
            integral_1=np.dot(boundary_node_1_vals,weights)*side_detJ*self.source_strength
            integral_2=np.dot(boundary_node_2_vals,weights)*side_detJ*self.source_strength
            self.F[side_nodes[0]]+=integral_1
            self.F[side_nodes[1]]+=integral_2 
         
        """
        Right
        """
          
        #Determinates of mappings between segment and parent element for each element
        x_vals=self.x[self.global_nodes_right]
        y_vals=self.y[self.global_nodes_right]
        
        detJ_right=np.sqrt((y_vals[:,1]-y_vals[:,0])**2+(x_vals[:,1]-x_vals[:,0])**2)/2

        for side_num in range(int(len(self.elements_right))):
            side_nodes=self.global_nodes_right[side_num]
            side_detJ=detJ_right[side_num] #constant on each side
            integral_1=np.dot(boundary_node_1_vals,weights)*side_detJ*self.source_strength
            integral_2=np.dot(boundary_node_2_vals,weights)*side_detJ*self.source_strength
            self.F[side_nodes[0]]+=integral_1
            self.F[side_nodes[1]]+=integral_2  
       
        if test==False:
            return self.F
        else:
            return x_vals,y_vals,detJ_bottom*2,np.hstack((detJ_left,detJ_top,detJ_right,detJ_bottom))
        
        
    def interface_cond(self,test=False):
        """
        Applies conditions to the interface. Tailoredd for linears currently
        """
        self.assemble()
        #Integration points along each element side
        pts_xi,weights=self.compute_gauss_points_and_weights(2,1)   
        
        #Generates basis function values for each integration point
        filler=np.ones(len(pts_xi))*-1 #arbitarry vector used to generate characteristic values for each basis function at int. points
        
        boundary_node_1_vals=self.N(pts_xi,filler,self.degree)[0,:] #values for first node in element side
        boundary_node_2_vals=self.N(pts_xi,filler,self.degree)[1,:] #"" "" second node
        
        global_nodes_in_sideset=self.global_nodes_in_sideset
        
        #Determinates of mappings between segment and parent element for each element
        x_vals=self.x[global_nodes_in_sideset]
        y_vals=self.y[global_nodes_in_sideset]
        
        detJ=np.sqrt((y_vals[:,1]-y_vals[:,0])**2+(x_vals[:,1]-x_vals[:,0])**2)/2
        
#        source_strength_per_l=self.source_strength
#        source_strength_per_l=self.source_strength/(2*np.pi)   len(self.elements_in_sideset)*detJ[0]*2
        if test==True:
            return detJ,self.elements_in_sideset,np.dot(boundary_node_1_vals,weights)
            
        for side_num in range(int(len(self.elements_in_sideset))):
            side_nodes=self.global_nodes_in_sideset[side_num]
            side_detJ=detJ[side_num] #constant on each side
            integral_1=np.dot(boundary_node_1_vals,weights)*side_detJ*self.source_strength
            integral_2=np.dot(boundary_node_2_vals,weights)*side_detJ*self.source_strength
            self.F[side_nodes[0]]+=integral_1
            self.F[side_nodes[1]]+=integral_2
         
        if test==True:
            return detJ,self.elements_in_sideset
#        else:
#            return detJ[0]*2,len(self.elements_in_sideset)
        
    def bound_cond(self):
        
        """
        Applies boundary conditions
        """  
        
##        Info for essential conditions
#        essential_nodes=self.essential_nodes #refere to indeces in K and F
#        essential_node_vals=self.essential_node_vals
#              
##        Dirichlet conditions
#        self.F+= np.einsum('ij,j',-self.K[:,essential_nodes],essential_node_vals) #corrects F for known nodal values
#        self.K[essential_nodes,:]=0
#        self.K[:,essential_nodes]=0
#        self.K[essential_nodes,essential_nodes]=1.
#        self.F[essential_nodes]=essential_node_vals
        """
        """
        
        
#        ess_node_corners=self.essential_nodes_top_bottom
#        ess_node_corners=self.essential_nodes_left_right
#        corner_vals=np.array([1,1,1,1])*0
        
#        ess_node_corners=self.essential_nodes_corners_and_interior
#        ess_node_corners=self.essential_nodes_interior

#        ess_node_corners= np.unique(self.global_nodes_in_sideset)
#        corner_vals=np.array([-1,-1,1,1])*40.*10**6
        """
        Verification
        """
#        def p_top(xvec,w):
#            return w*np.sin(np.pi*(400+xvec)/1600)
#        ess_node_corners=np.hstack(( self.nodes_left-1, self.nodes_bottom-1, self.nodes_top-1))
##        print(ess_node_corners)
#        corner_vals=np.hstack((np.zeros(len(self.nodes_left)),np.zeros(len(self.nodes_bottom)), p_top(self.x[self.nodes_top-1],10**7)))
##        print(corner_vals)        
        """
        Benchmark problem
        """
        ess_node_corners= self.essential_nodes_corners
        corner_vals=np.ones(len(ess_node_corners))*40.*10**6
#        corner_vals=np.ones(len(ess_node_corners))*0


        
        
#        print(ess_node_corners,corner_vals)
        ###Dirichlet conditions
        self.F+= np.einsum('ij,j',-self.K[:,ess_node_corners],corner_vals) #corrects F for known nodal values
        self.K[ess_node_corners,:]=0
        self.K[:,ess_node_corners]=0
        self.K[ess_node_corners,ess_node_corners]=1.
        self.F[ess_node_corners]=corner_vals
        
#        nat_nodes= np.unique(self.global_nodes_in_sideset)
#        nat_nodes= self.essential_nodes_corners
#        nat_nodes= self.essential_nodes_interior
        
#        nat_nodes=self.essential_nodes_corners_and_interior
#        self.F[nat_nodes]+=np.ones(len(nat_nodes))*self.source_strength
        
    def nodal_values(self):
        """
        Solves system to find approximation at nodes
        """
#        self.assemble()
        self.interface_cond()
#        self.edge_integral()
        self.bound_cond()
        
        K_s=csr_matrix(self.K.tolist())
        self.approx=spsolve(K_s,self.F)/10**6
      
        
    def plotter(self,mesh=True):
        """
        Plots solution and overlays mesh
        """           
#        approx=self.nodal_values()
        
        xy = np.array([self.x[:], self.y[:]]).T
        patches = []
        for coords in xy[self.con_array[:]]:
            quad = Polygon(coords,facecolor='none',fill=False)
            patches.append(quad)
        
        n=50
        X,Y=np.meshgrid(np.linspace(min(self.x),max(self.x),n),np.linspace(min(self.y),max(self.y),n))
#        X=X.ravel(); Y=Y.ravel()
#        Z=np.zeros(n**2)
#        print(X,Y)
#        for i in range(n**2):
#            print(self.approx_function(X[i],Y[i]),i)
#            Z[i]=self.approx_function(X[i],Y[i])
#        f=scipy.interpolate.interp2d(self.x,self.y,self.approx)
#        Z=f(np.linspace(min(self.x),max(self.x),n),np.linspace(min(self.y),max(self.y),n))
#        X=X.reshape((n,n)); Y=Y.reshape((n,n)); Z=Z.reshape((n,n))
        Z=scipy.interpolate.griddata((self.x,self.y),self.approx,(X,Y))
        ticks=np.linspace(min(self.approx),max(self.approx),5)
#        return (X,Y,Z.astype('float'))
        fig,ax=plt.subplots()

        cs=ax.contourf(X,Y,Z,cmap="coolwarm",levels=np.linspace(min(self.approx),max(self.approx),30)) 
        cbar=fig.colorbar(cs,ax=ax,ticks=ticks)
        
        
        p = PatchCollection(patches, match_original=True)
        p.set_linewidth(0.1)

        if mesh==False:
            
            p.set_linewidth(0)
        
        plt.xticks(np.linspace(-400,400,5))
        plt.yticks(np.linspace(-400,400,5))
        ax.add_collection(p)
        ax.set_aspect('equal') 
        ax.set_xlim([-400, 400])
        ax.set_ylim([-400, 400])
        props = dict(boxstyle='square', facecolor='white', alpha=1.)
        plt.gcf().subplots_adjust(bottom=0.15)   
        plt.text(0.5, 0.9,'Degrees of freedom= %s' %len(self.F), ha='center', va='center', transform=ax.transAxes,bbox=props,fontsize=14)
        plt.xlabel('x [m]',fontsize=18)
        plt.ylabel('y [m]',fontsize=18)
        plt.tick_params(labelsize=18)
        cbar.ax.tick_params(labelsize=18)
        cbar.set_label('Pressure [MPa]', labelpad=10, size=18)
        
        return plt
    
    def mapping(self,element,xi,eta):
        """
        Maps from parent space to physical space given an element number and xi,eta pair
        """
        elem_con=self.con_array[element]
        x_ele=self.x[elem_con]
        y_ele=self.y[elem_con]
        
#        return np.vstack((x_ele,y_ele)).T
        coeff_array=np.dot(np.array([[1,-1,1,-1],[-1,1,1,-1],[-1,-1,1,1]]),np.vstack((x_ele,y_ele)).T)
        x=(coeff_array[0,0]*xi*eta+coeff_array[1,0]*xi+coeff_array[2,0]*eta+np.sum(x_ele))/4.
        y=(coeff_array[0,1]*xi*eta+coeff_array[1,1]*xi+coeff_array[2,1]*eta+np.sum(y_ele))/4.
        
        return x,y
        
    def inverse_mapping(self,element,x,y):
        """
        Maps from physical space to parent space. Based off of work in Hua, 1990
        """
        elem_con=self.con_array[element]
        x_ele=self.x[elem_con]
        y_ele=self.y[elem_con]
        
        coeff_array=np.dot(np.array([[1,-1,1,-1],[-1,1,1,-1],[-1,-1,1,1]]),np.vstack((x_ele,y_ele)).T)
        
        a1,a2=coeff_array[0,:]
        b1,b2=coeff_array[1,:]
        c1,c2=coeff_array[2,:]
        d1,d2=4.*x-np.sum(x_ele),4.*y-np.sum(y_ele)
        
        if a1==0:
            if a2==0:
                xi=(d1*c2-d2*c1)/(b1*c2-b2*c1); eta=(b1*d2-b2*d1)/(b1*c2-b2*c1)
            elif a2!=0:
                if c1==0:
                    xi=d1/b1; eta=(b1*d2-b2*d1)/(a2*d1+b1*c2)
                elif c2!=0:
                    #Defining coeffs in quadratic equation
                    a=a2*b1; b=c2*b1-a2*d1-b2*c1; c=d2*c1-c2*d1
                    xi=np.array([  (-b+np.sqrt(b**2-4*a*c))/(2*a) , (-b-np.sqrt(b**2-4*a*c))/(2*a)     ])
                   
                    xi_index=np.where(abs(xi)<=1.01)
                    xi=xi[xi_index]
                    
                    eta=(d1-b1*xi)/c1
                    
        elif a1!=0:
            if a2!=0:
                if a1*b2-a2*b1!=0:
                    if a1*c2-a2*c1!=0:
                        #Defining coeffs in quadratic equation
                        a=a1*(a1*b2-a2*b1); b=c1*(a1*b2-a2*b1)-a1*(a1*d2-a2*d1)-b1*(a1*c2-a2*c1); c=d1*(a1*c2-a2*c1)-c1*(a1*d2-a2*d1)
                        xi=np.array([(-b+np.sqrt(b**2-4*a*c))/(2*a) , (-b-np.sqrt(b**2-4*a*c))/(2*a)])
                        
                        xi_index=np.where(abs(xi)<=1.01) #NEED TO FIND OUT WHY THIS IS HAPPENING
                        xi=xi[xi_index]
                        
                        eta=((a1*d2-a2*d1)+(b1*a2-b2*a1)*xi)/(a1*c2-a2*c1)
                  
                    elif a1*c2-a2*c1==0:
                        xi=(a1*d2-a2*d1)/(a1*b2-a2*b1)
                        eta=a1*(d1*b2-d2*b1)/(c1*(a1*b2-a2*b1)+a1*(a1*d2-a2*d1))
              
                elif a1*b2-a2*b1==0:
                    xi=a1*(d1*c2-d2*c1)/(b1*(a1*c2-a2*c1)+a1*(a1*d2-a2*d1))
                    eta=(a1*d2-a2*d1)/(a1*c2-a2*c1)
            elif a2==0:
                if b2==0:
                    xi=(d1*c2-d2*c1)/(a1*d2+b1*c2); eta=d2/c2
                elif b2!=0:
                    #Defining coeffs in quadratic equation
                    a=a1*b2; b=c1*b2-a1*d2-b1*c2; c=d1*c2-c1*d2
                    xi=np.array([  (-b+np.sqrt(b**2-4*a*c))/(2*a) , (-b-np.sqrt(b**2-4*a*c))/(2*a)     ])
                    
                    xi_index=np.where(abs(xi)<=1.01)
                    xi=xi[xi_index]
                    
                    eta=(d2-b2*xi)/c2
                    
        return xi,eta
                 
    def approx_function(self,test_x,test_y,test=False):
        """
        Computes the error using the current refinement against a very fine mesh. Doesn't seem to work for points along bottom boundary
        """
        
        #Computes distance from each node to test point
        distance_array=np.sqrt((self.x-test_x)**2+(self.y-test_y)**2)
        
        #Finds index of closest node to test point
        closest_pt_index=distance_array.argmin()
                
        #Finds elements sharing closest node
        test_elements=np.where(self.con_array==closest_pt_index)[0]
        
        #Finds which element contains test point
        for i in range(len(test_elements)):
            ele_num=test_elements[i]
            ele_con=self.con_array[ele_num,:]
            poly_corners=np.vstack((self.x[ele_con],self.y[ele_con])).T
#            poly_corners=np.vstack((poly_corners,poly_corners[0,:]))
            bbPath = mplPath.Path(poly_corners)
            if bbPath.contains_point((test_x, test_y)) or (test_y==-400 and test_x<=np.max(poly_corners[:,0]) and test_x>=np.min(poly_corners[:,0])): #TEMPORARY FIX
                break
            elif i==len(test_elements)-1:
#                print "No valid element found"
                return -1,poly_corners
     
        #Finds solution at element nodes
        nodal_vals=self.approx[ele_con].transpose()

        #Find xi,eta location with respect to element
        xi,eta=self.inverse_mapping(ele_num,test_x,test_y)
#        xi=xi[0];eta=eta[0]

        value_approx=np.dot(nodal_vals,self.N(xi,eta,1))
        
        if test==True:
            return poly_corners,ele_con,ele_num
        else:
            #Finds approximation
            return value_approx


        
        
        
"""
Robin and dirichlet
"""
#degree=1
#  
#def k_term(x,y):
#    
#    return np.ones(len(x))
#    
#def f_term(x,y):
#    
#    return np.ones(len(x))*0
#    
#bc_top=lambda x,y: 0
#bc_bottom=lambda x,y: 0
#bc_left=lambda x,y: 0
#bc_right=lambda x,y: 0
#
#bc_array=lambda x,y: np.array([bc_top(x,y),bc_bottom(x,y),bc_left(x,y),bc_right(x,y)])
#           
#demo_2=SFEM_2D(degree,'Data-selected/s_curve_mi_2.g',k_term,f_term,np.array([10.,0,0,0]),-10.) 
#demo_2.plotter()
##
#demo_4=SFEM_2D(degree,'Data-selected/s_curve_mi_4.g',k_term,f_term,np.array([10.,0,0,0]),-10.) 
#demo_4.plotter()
##
#demo_8=SFEM_2D(degree,'Data-selected/s_curve_mi_8.g',k_term,f_term,np.array([10.,0,0,0]),-10.) 
#demo_8.plotter()
##
#demo_16=SFEM_2D(degree,'Data-selected/s_curve_mi_16.g',k_term,f_term,np.array([10.,0,0,0]),-10.) 
#demo_16.plotter()
##
#demo_32=SFEM_2D(degree,'Data-selected/s_curve_mi_32.g',k_term,f_term,np.array([10.,0,0,0]),-10.) 
#demo_32.plotter()
##
#demo_64=SFEM_2D(degree,'Data-selected/s_curve_mi_64.g',k_term,f_term,np.array([10.,0,0,0]),-10.) 
#demo_64.plotter()

#demo_127=SFEM_2D(degree,'Data-selected/s_curve_mi_127.g',k_term,f_term,np.array([10.,0,0,0]),-10.) 
#demo_127.plotter()

#demo_256=SFEM_2D(degree,'Data-selected/s_curve_mi_256.g',k_term,f_term,np.array([0.,0,0,0]),-10.) 

#def L2_error(object2, object1,test_x,test_y):
#    """
#    Finds L2 error between 2 meshes using node locations of finer mesh as test points. Object 2 assumed to give the better solution
#    """
#    x=test_x
#    y=test_y
##    x=object2.x
##    y=object2.y
#    
#    flag_count=0
#    for i in range(len(x)):
#        if i==0:
#            ob_1_approx=object1.approx_function(x[i],y[i])
#            ob_2_approx=object2.approx_function(x[i],y[i])
#            if ob_1_approx==-1 or ob_2_approx==-1:
#                flag_count+=1
#                error=np.empty(0)
#                x_flag=x[i]
#                y_flag=y[i]
#                continue
#            else:
#                error=np.array([(object2.approx_function(x[i],y[i])-object1.approx_function(x[i],y[i]))**2])
#                x_flag=np.empty(0)
#                y_flag=np.empty(0)
#        else:
#            ob_1_approx=object1.approx_function(x[i],y[i])
#            ob_2_approx=object2.approx_function(x[i],y[i])
#            if ob_1_approx==-1 or ob_2_approx==-1:
#                x_flag_i=x[i]; y_flag_i=y[i]
#                x_flag=np.append(x_flag,x_flag_i)
#                y_flag=np.append(y_flag,y_flag_i)
#                
#                flag_count+=1
#                continue
#            else:
#            
#                err=np.array([(object2.approx_function(x[i],y[i])-object1.approx_function(x[i],y[i]))**2])
#                error=np.append(error,err)
#        
#    return np.sqrt(np.sum(error)),x_flag,y_flag
#
#def flag_check(object1,object2):
#    """
#    Check to see where valid element isn't found
#    """
#    
#    flag=0
#    debug=0
#    iarr=0
#    for i in range(len(object2.x)):
#        if i==0:
#            a=object1.approx_function(object2.x[i],object2.y[i])
#            if a==-1:
#                flag+=1
#                x_flag=object2.x[i]
#                y_flag=object2.y[i]
#                continue
#            x_flag=np.empty(0)
#            y_flag=np.empty(0)
#            dubug=np.array([a])
#            iarr=np.array([i])
#        else:
#            a=object1.approx_function(object2.x[i],object2.y[i])
#            if a==-1:
#                flag+=1
#                x_flag_i=object2.x[i]; x_flag=np.append(x_flag,x_flag_i)
#                y_flag_i=object2.y[i]; y_flag=np.append(y_flag,y_flag_i)
#                continue
#            debug=np.append(debug,a)
#            iarr=np.append(iarr,i)
#            
#        return debug,iarr,x_flag,y_flag
        

#L2_2,x_flag_2,y_flag_2=L2_error(demo_64,demo_2)
#L2_4,x_flag_4,y_flag_4=L2_error(demo_64,demo_4)
#L2_8,x_flag_8,y_flag_8=L2_error(demo_64,demo_8)
#L2_16,x_flag_16,y_flag_16=L2_error(demo_64,demo_16)
#L2_32,x_flag_32,y_flag_32=L2_error(demo_64,demo_32)       
        
#        
#L2_2,x_flag_2,y_flag_2=L2_error(demo_64,demo_2,test_x,test_y)
#L2_4,x_flag_4,y_flag_4=L2_error(demo_64,demo_4,test_x,test_y)
#L2_8,x_flag_8,y_flag_8=L2_error(demo_64,demo_8,test_x,test_y)
#L2_16,x_flag_16,y_flag_16=L2_error(demo_64,demo_16,test_x,test_y)
#L2_32,x_flag_32,y_flag_32=L2_error(demo_64,demo_32,test_x,test_y)

#plt.plot(np.log2([1,.5,.25,1/8.,1/16.]),  \
#    np.log2([L2_2,L2_4,L2_8,L2_16,L2_32]),'-o')
#plt.xlabel('Element Length Scale')
#plt.ylabel('L2 Error')
#plt.title('loglog Plot (base 2) of Error vs. Refinement')



