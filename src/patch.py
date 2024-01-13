# Copyright Arthur Leclaire (c), 2019.
#  https://plmlab.math.cnrs.fr/aleclaire/texto

#  Modified by Clément Weinreich and Paul-Henri Pinart in 2023

import numpy as np

class patch:

    def __init__(self,m,n,nc,w,s=1):
        # Initialize instance of class patch.
        #  P = patch(m,n,nc,w,s=1)
        #
        #  INPUT
        #   m,n    Spatial dimensions
        #   nc     Number of channels
        #   w      Patch size (w x w)
        #   s      Stride
        
        self.m = m
        self.n = n
        self.nc = nc
        self.w = w
        self.stride = s
        self.pdim = w*w*nc
        [x,y] = np.mgrid[0:m-w+1:s,0:n-w+1:s]
        Np = x.shape[0]*x.shape[1]  # number of patches
        self.Np = Np
        [dx,dy] = np.mgrid[0:w,0:w]
        X = x[:,:,np.newaxis,np.newaxis]+dx[np.newaxis,np.newaxis,:]
        Y = y[:,:,np.newaxis,np.newaxis]+dy[np.newaxis,np.newaxis,:]
        px = X.reshape((Np,w*w))
        py = Y.reshape((Np,w*w))
        px = np.tile(px[:,:,np.newaxis],(1,nc))
        py = np.tile(py[:,:,np.newaxis],(1,nc))
        px = px.reshape(Np,self.pdim)
        py = py.reshape(Np,self.pdim)
        pc = np.tile(np.arange(0,nc),(w*w,Np))
        self.pc = pc.reshape((Np,w*w*nc))
        #self.pc = pc.T
        self.px = px
        self.py = py

    def im2patch(self,u):
        # Extract patches from image u.
        #  Pu = P.im2patch(u)
        
        pu = u[self.px,self.py,self.pc]
        return pu
    
        
    def patch2im(self,p,weight=None):
        if(weight is None):
            weight = np.zeros(p.shape[0])
        else:
            weight = 8*(weight - weight.min())/(weight.max()-weight.min())-4
            weight = 1/(1+np.exp(-weight))

        m,n,nc = self.m,self.n,self.nc
        px,py,pc = self.px,self.py,self.pc
        u = np.zeros((m,n,nc))
        z = np.zeros((m,n,nc))
        for j in range(0,self.pdim):
            u[px[:,j],py[:,j],pc[:,j]] += p[:,j] * (1-weight)
            z[px[:,j],py[:,j],pc[:,j]] += (1-weight)
        
        u = u/z
        return u


    def patch2im_median(self, p):
        m, n, nc = self.m, self.n, self.nc
        px, py, pc = self.px, self.py, self.pc
        pdim = self.pdim

        # Initialize a 3D list to store pixel values
        # Each channel has a separate 2D list
        pixel_values = [[[] for _ in range(n)] for _ in range(m)]
        for c in range(nc):
            pixel_values[c] = [[[] for _ in range(n)] for _ in range(m)]

        # Accumulate pixel values from each patch
        for j in range(pdim):
            for i in range(len(px[:, j])):
                x, y, c = px[i, j], py[i, j], pc[i, j]
                pixel_values[c][x][y].append(p[i, j])

        # Create an empty array for the final image
        u = np.zeros((m, n, nc))

        # Compute median for each pixel
        for c in range(nc):
            for i in range(m):
                for j in range(n):
                    if pixel_values[c][i][j]:
                        u[i, j, c] = np.median(pixel_values[c][i][j])
        return u
