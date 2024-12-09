# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 14:23:25 2018

@author: nsde
"""

#%%
import os, pickle
import numpy as np
import scipy.linalg as la

def make_hashable(arr):
    """ Make an array hasable. In this way we can use built-in functions like
        set(...) and intersection(...) on the array
    """
    return tuple([tuple(r.tolist()) for r in arr])

#%%
class Tesselation(object):
    """ Base tesselation class. This function is not meant to be called,
        but descripes the base structure that needs to be implemented in
        1D, 2D, and 3D. Additionally, some functionallity is shared across
        the different dimensions.
        
    Args:
        nc: list with number of cells
        domain_min: value of the lower bound(s) of the domain
        domain_max: value of the upper bound(s) of the domain
        zero_boundary: bool, if true the velocity is zero on the boundary
        volume_perservation: bool, if true volume is perserved
        
    Methods that should not be implemented in subclasses:
        @get_cell_centers:
        @create_continuity_constrains:
        @create_zero_trace_constrains:
            
    Methods that should be implemented in subclasses:
        @find_verts:
        @find_verts_outside:
        @create_zero_boundary_constrains:
        
    """
    def __init__(self, nc, domain_min, domain_max,
                 zero_boundary = True, volume_perservation=False, 
                 direc=None, override=False):
        """ Initilization of the class that create the constrain matrix L
        Arguments:
            nc: list, number of cells in each dimension
            domain_min: list, lower domain bound in each dimension
            domain_max: list, upper domain bound in each dimension
            zero_boundary: bool, determines is the velocity at the boundary is zero
            volume_perservation: bool, determine if the transformation is
                volume perservating
            direc: string, where to store the basis
            override: bool, determines if we should calculate the basis even
                if it already exists
        """
        
        # Save parameters
        self.nc = nc
        self.domain_min = domain_min
        self.domain_max = domain_max
        self.zero_boundary = zero_boundary
        self.volume_perservation = volume_perservation
        self.dir = direc
        self._basis_file = self.dir + \
                            'cpab_basis_dim' + str(len(self.nc)) + '_tess' + \
                            '_'.join([str(e) for e in self.nc]) + '_' + \
                            'vo' + str(int(not self.zero_boundary)) + '_' + \
                            'zb' + str(int(self.zero_boundary)) + '_' + \
                            'vp' + str(int(self.volume_perservation))

        # Check if file exist else calculate the basis
        if not os.path.isfile(self._basis_file+'.cpab_basis') or override:
            # Get vertices
            self.find_verts()
            
            # Find shared vertices
            self.find_shared_verts()
            
            # find auxility vertices, if transformation is valid outside
            if not zero_boundary: self.find_verts_outside()
            
            # Get continuity constrains
            self.L = self.create_continuity_constrains()
            
            # If zero boundary, add constrains
            if zero_boundary:
                temp = self.create_zero_boundary_constrains()
                self.L = np.concatenate((self.L, temp), axis=0)
                
            # If volume perservation, add constrains
            if volume_perservation:
                temp = self.create_zero_trace_constrains()
                self.L = np.concatenate((self.L, temp), axis=0)
            
            # Find null space
            u, s, vh = la.svd(self.L)
            padding = np.max([0, np.shape(self.L)[-1] - np.shape(s)[0]])
            null_mask = np.concatenate(((s <= 1e-6), np.ones((padding,), dtype=bool)), axis=0)
            null_space = np.compress(null_mask, vh, axis=0)
            self.B = np.transpose(null_space)
        
            # Save to file
            with open(self._basis_file + '.cpab_basis', 'wb') as f:
                pickle.dump(self.__dict__, f, pickle.HIGHEST_PROTOCOL)
        
        else:
            with open(self._basis_file + '.cpab_basis', 'rb') as f:
                self.__dict__ = pickle.load(f)
    
    def get_cell_centers(self):
        """ Get the centers of all the cells """
        return np.mean(self.verts[:,:,:self.ndim], axis=1)
    
    def find_verts(self):
        """ Function that should find the different vertices of all cells in
            the tesselation """
        raise NotImplementedError
        
    def find_shared_verts(self):
        """ Find pairs of cells that share ndim-vertices. It is these pairs,
            where we need to add continuity constrains at """
        # Iterate over all pairs of cell to find cells with intersecting cells
        shared_v, shared_v_idx = [ ], [ ]
        for i in range(self.nC):
            for j in range(self.nC):
                if i != j:
                    vi = make_hashable(self.verts[i])
                    vj = make_hashable(self.verts[j])
                    shared_verts = set(vi).intersection(vj)
                    if len(shared_verts) == self.ndim and (j,i) not in shared_v_idx:
                        shared_v.append(list(shared_verts)[:self.ndim])
                        shared_v_idx.append((i,j))
        
        # Save result
        self.shared_v = np.asarray(shared_v)
        self.shared_v_idx = shared_v_idx
        
    def find_verts_outside(self):
        """ If the transformation should be valid outside, this function should
            add additional auxilliry points to the tesselation that secures
            continuity outside the domain """
        raise NotImplementedError
        
    def create_continuity_constrains(self):
        """ This function goes through all pairs (i,j) of cells that share a
            boundary. In N dimension we need to add N*N constrains (one for each
            dimension times one of each vertex in the boundary) """
        Ltemp = np.zeros(shape=(0,self.n_params*self.nC))
        for idx, (i,j) in enumerate(self.shared_v_idx):
            for vidx in range(self.ndim):
                for k in range(self.ndim):
                    index1 = self.n_params*i + k*(self.ndim+1)
                    index2 = self.n_params*j + k*(self.ndim+1)
                    row = np.zeros(shape=(1,self.n_params*self.nC))
                    row[0,index1:index1+(self.ndim+1)] = self.shared_v[idx][vidx]
                    row[0,index2:index2+(self.ndim+1)] = -self.shared_v[idx][vidx]
                    Ltemp = np.vstack((Ltemp, row))
        return Ltemp
        
    def create_zero_boundary_constrains(self):
        """ Function that creates a constrain matrix L, containing constrains that
            secure 0 velocity at the boundary """
        raise NotImplementedError
        
    def create_zero_trace_constrains(self):
        """ The volume perservation constrains, that corresponds to the trace
            of each matrix being 0. These can be written general for all dims."""
        Ltemp = np.zeros((self.nC, self.n_params*self.nC))
        row = np.concatenate((np.eye(self.ndim), np.zeros((self.ndim, 1))), axis=1).flatten()
        for c in range(self.nC):
            Ltemp[c,self.n_params*c:self.n_params*(c+1)] = row
        return Ltemp
        
#%%
class Tesselation1D(Tesselation):
    def __init__(self, nc, domain_min, domain_max,
                 zero_boundary = True, volume_perservation=False, 
                 direc=None, override=False):
        # 1D parameters
        self.n_params = 2
        self.nC = np.prod(nc)
        self.ndim = 1
        
        # Initialize super class
        super(Tesselation1D, self).__init__(nc, domain_min, domain_max,
             zero_boundary, volume_perservation, direc, override)
        
    def find_verts(self):
        Vx = np.linspace(self.domain_min[0], self.domain_max[0], self.nc[0]+1)
        
        # Find cell index and verts for each cell
        cells, verts = [ ], [ ]
        for i in range(self.nc[0]):
            v1 = tuple([Vx[i], 1])
            v2 = tuple([Vx[i+1], 1])
            verts.append((v1, v2))
            cells.append((i))
        
        # Convert to array
        self.verts = np.asarray(verts)
        self.cells = cells
        
    def find_verts_outside(self):
        pass # in 1D, we do not need auxilliry points
        
    def create_zero_boundary_constrains(self):
        Ltemp = np.zeros((2,2*self.nC))
        Ltemp[0,:2] = [self.domain_min[0], 1]
        Ltemp[1,-2:] = [self.domain_max[0], 1]
        return Ltemp

#%%
class Tesselation2D(Tesselation):
    def __init__(self, nc, domain_min, domain_max,
                 zero_boundary = True, volume_perservation=False, 
                 direc=None, override=False):
        # 1D parameters
        self.n_params = 6 
        self.nC = 4*np.prod(nc) # 4 triangle per cell
        self.ndim = 2
        
        # Initialize super class
        super(Tesselation2D, self).__init__(nc, domain_min, domain_max,
             zero_boundary, volume_perservation, direc, override)
    
    def find_verts(self):
        Vx = np.linspace(self.domain_min[0], self.domain_max[0], self.nc[0]+1)
        Vy = np.linspace(self.domain_min[1], self.domain_max[1], self.nc[1]+1)
        
        # Find cell index and verts for each cell
        cells, verts = [ ], [ ]
        for i in range(self.nc[1]):
            for j in range(self.nc[0]):
                ul = tuple([Vx[j],Vy[i],1])
                ur = tuple([Vx[j+1],Vy[i],1])
                ll = tuple([Vx[j],Vy[i+1],1])
                lr = tuple([Vx[j+1],Vy[i+1],1])
                
                center = [(Vx[j]+Vx[j+1])/2,(Vy[i]+Vy[i+1])/2,1]
                center = tuple(center)                 
                
                verts.append((center,ul,ur))  # order matters!
                verts.append((center,ur,lr))  # order matters!
                verts.append((center,lr,ll))  # order matters!
                verts.append((center,ll,ul))  # order matters!                
        
                cells.append((j,i,0))
                cells.append((j,i,1))
                cells.append((j,i,2))
                cells.append((j,i,3))
                
        # Convert to array
        self.verts = np.asarray(verts)
        self.cells = cells
        
    def find_verts_outside(self):
        shared_v, shared_v_idx = [ ], [ ]
        
        left =   np.zeros((self.nC, self.nC), np.bool)    
        right =  np.zeros((self.nC, self.nC), np.bool) 
        top =    np.zeros((self.nC, self.nC), np.bool) 
        bottom = np.zeros((self.nC, self.nC), np.bool) 
        
        for i in range(self.nC):
            for j in range(self.nC):
                
                vi = make_hashable(self.verts[i])
                vj = make_hashable(self.verts[j])
                shared_verts = set(vi).intersection(vj)
                
                mi = self.cells[i]
                mj = self.cells[j]
        
                # leftmost col, left triangle, adjacent rows
                if  mi[0]==mj[0]==0 and \
                    mi[2]==mj[2]==3 and \
                    np.abs(mi[1]-mj[1])==1: 
                        
                    left[i,j]=True
                
                # rightmost col, right triangle, adjacent rows                 
                if  mi[0]==mj[0]==self.nc[0]-1 and \
                    mi[2]==mj[2]==1 and \
                    np.abs(mi[1]-mj[1])==1: 
        
                    right[i,j]=True
                
                # uppermost row, upper triangle , adjacent cols                    
                if  mi[1]==mj[1]==0 and \
                    mi[2]==mj[2]==0 and \
                    np.abs(mi[0]-mj[0])==1:
                        
                    top[i,j]=True
                
                # lowermost row, # lower triangle, # adjacent cols            
                if  mi[1]==mj[1]==self.nc[1]-1 and \
                    mi[2]==mj[2]==2 and \
                    np.abs(mi[0]-mj[0])==1:
                        
                    bottom[i,j]=True
                                
                if  len(shared_verts) == 1 and \
                    any([left[i,j],right[i,j],top[i,j],bottom[i,j]]) and \
                    (j,i) not in shared_v_idx:
                        
                    v_aux = list(shared_verts)[0] # v_aux is a tuple
                    v_aux = list(v_aux) # Now v_aux is a list (i.e. mutable)
                    if left[i,j] or right[i,j]:
                        v_aux[0]-=10 # Create a new vertex  with the same y
                    elif top[i,j] or bottom[i,j]:
                        v_aux[1]-=10 # Create a new vertex  with the same x
                    else:
                        raise ValueError("WTF?")                        
                    shared_verts = [tuple(shared_verts)[0], tuple(v_aux)]
                    shared_v.append(shared_verts)
                    shared_v_idx.append((i,j))
        
        # Concat to the current list of vertices
        if shared_v:
            self.shared_v = np.concatenate((self.shared_v, shared_v))
            self.shared_v_idx = np.concatenate((self.shared_v_idx, shared_v_idx))
        
    def create_zero_boundary_constrains(self):
        xmin, ymin = self.domain_min
        xmax, ymax = self.domain_max
        Ltemp = np.zeros(shape=(0,6*self.nC))
        for c in range(self.nC):
            for v in self.verts[c]:
                if(v[0] == xmin or v[0] == xmax): 
                    row = np.zeros(shape=(6*self.nC))
                    row[(6*c):(6*(c+1))] = np.append(v,np.zeros((1,3)))
                    Ltemp = np.vstack((Ltemp, row))
                if(v[1] == ymin or v[1] == ymax): 
                    row = np.zeros(shape=(6*self.nC))
                    row[(6*c):(6*(c+1))] = np.append(np.zeros((1,3)),v)
                    Ltemp = np.vstack((Ltemp, row))
        return Ltemp

#%%
class Tesselation3D(Tesselation):
    def __init__(self, nc, domain_min, domain_max,
                 zero_boundary = True, volume_perservation=False, 
                 direc=None, override=False):
        # 1D parameters
        self.n_params = 12
        self.nC = 5*np.prod(nc) # 6 triangle per cell
        self.ndim = 3
        
        # Initialize super class
        super(Tesselation3D, self).__init__(nc, domain_min, domain_max,
             zero_boundary, volume_perservation, direc, override)
    
    def find_verts(self):
        Vx = np.linspace(self.domain_min[0], self.domain_max[0], self.nc[0]+1)
        Vy = np.linspace(self.domain_min[1], self.domain_max[1], self.nc[1]+1)
        Vz = np.linspace(self.domain_min[2], self.domain_max[2], self.nc[2]+1)
        
        # Find cell index and verts for each cell
        cells, verts = [ ], [ ]
        for i in range(self.nc[2]):
            for j in range(self.nc[1]):        
                for k in range(self.nc[0]):
                    ul0 = tuple([Vx[k],Vy[j],Vz[i],1])
                    ur0 = tuple([Vx[k+1],Vy[j],Vz[i],1])
                    ll0 = tuple([Vx[k],Vy[j+1],Vz[i],1])
                    lr0 = tuple([Vx[k+1],Vy[j+1],Vz[i],1])
                    ul1 = tuple([Vx[k],Vy[j],Vz[i+1],1])
                    ur1 = tuple([Vx[k+1],Vy[j],Vz[i+1],1])
                    ll1 = tuple([Vx[k],Vy[j+1],Vz[i+1],1])
                    lr1 = tuple([Vx[k+1],Vy[j+1],Vz[i+1],1])

                    tf=False                    
                    if k%2==0:
                        if (i%2==0 and j%2==1) or  (i%2==1 and j%2==0):
                            tf=True
                    else:
                        if (i%2==0 and j%2==0) or  (i%2==1 and j%2==1):
                            tf=True
                    
                    if tf:
                        ul0,ur0,lr0,ll0 = ur0,lr0,ll0,ul0
                        ul1,ur1,lr1,ll1 = ur1,lr1,ll1,ul1
                    
                    # ORDER MATTERS 
                    verts.append((ll1,ur1,ul0,lr0))  # central part
                    verts.append((ul1,ur1,ll1,ul0))
                    verts.append((lr1,ur1,ll1,lr0))
                    verts.append((ll0,ul0,lr0,ll1))
                    verts.append((ur0,ul0,lr0,ur1))
                    
                    for l in range(5):
                        cells.append((k,j,i,l))
        
        # Convert to array
        self.verts = np.asarray(verts)
        self.cells = cells

    def find_verts_outside(self):
        shared_verts, shared_verts_idx = [ ], [ ]
        # Iterate over all pairs of cells
        for i in range(self.nC):
            for j in range(self.nC):
                if i != j:
                    # Add constrains for each side
                    for d in range(self.ndim):
                        # Get cell vertices
                        vi = self.verts[i]    
                        vj = self.verts[j]
                        ci = self.cells[i]
                        cj = self.cells[j]
                        # Conditions for adding a constrain
                        upper_cond = sum(vi[:,d]==self.domain_min[d]) == 3 and \
                                     sum(vj[:,d]==self.domain_min[d]) == 3
                        lower_cond = sum(vi[:,d]==self.domain_max[d]) == 3 and \
                                     sum(vj[:,d]==self.domain_max[d]) == 3
                        dist_cond = (sum([abs(i1-i2) for i1,i2 in zip(ci[:3], cj[:3])]) == 0) # same cell
                        idx_cond = (j,i) not in shared_verts_idx
                        if (upper_cond or lower_cond) and dist_cond and idx_cond:
                            # Find the shared points
                            vi = make_hashable(vi)
                            vj = make_hashable(vj)
                            sv = set(vi).intersection(vj)
                            center = [(v1 + v2) / 2.0 for v1, v2 in zip(vi[0], vj[0])]
                            center[d] += (-1) if upper_cond else (+1)
                            shared_verts.append(list(sv.union([tuple(center)])))
                            shared_verts_idx.append((i,j))
                            
        # Add to already found pairs
        if shared_verts:
            self.shared_v = np.concatenate((self.shared_v, np.asarray(shared_verts)))
            self.shared_v_idx += shared_verts_idx

            
    def create_zero_boundary_constrains(self):
        xmin, ymin, zmin = self.domain_min
        xmax, ymax, zmax = self.domain_max
        Ltemp = np.zeros(shape=(0, 12*self.nC))
        for c in range(self.nC):
            for v in self.verts[c]:
                if(v[0] == xmin or v[0] == xmax):
                    row = np.zeros(shape=(12*self.nC))
                    row[(12*c):(12*(c+1))] = np.concatenate([v, np.zeros((8,))])
                    Ltemp = np.vstack((Ltemp, row))
                if(v[1] == ymin or v[1] == ymax):
                    row = np.zeros(shape=(12*self.nC))
                    row[(12*c):(12*(c+1))] = np.concatenate([np.zeros((4,)), v, np.zeros((4,))])
                    Ltemp = np.vstack((Ltemp, row))
                if(v[2] == zmin or v[2] == zmax):
                    row = np.zeros(shape=(12*self.nC))
                    row[(12*c):(12*(c+1))] = np.concatenate([np.zeros((8,)), v])
                    Ltemp = np.vstack((Ltemp, row))
        return Ltemp
                    
#%%
if __name__ == "__main__":
    tess1 = Tesselation1D([5], [0], [1], zero_boundary=True, volume_perservation=True)
    tess2 = Tesselation2D([2,2], [0,0], [1,1], zero_boundary=False, volume_perservation=True)
    tess3 = Tesselation3D([2,2,2], [0,0,0], [1,1,1], zero_boundary=True, volume_perservation=False)
