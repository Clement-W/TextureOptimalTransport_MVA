#  Copyright Arthur Leclaire (c), 2019.

import numpy as np
import sklearn.mixture                # for EM
from skimage.transform import resize
from scipy.linalg import sqrtm
from src.patch import *
from src.gaussian_texture import *
import src.semidiscrete_ot as sdot
import matplotlib.pyplot as plt
import time

class model:
    def __init__(self,im0, w, nscales, ngmm, visu=False, s=1, niter=100000, C=1,mode="BASETEXTO"):
        # Texture synthesis with patch optimal transport
        #  This function initializes the texture model.
        #  It computes all the model parameters from a given exemplar image.
        #
        #   texmodel = model(im0,w,nscales,ngmm, visu = False, s=1, niter=100000, C=1)
        #
        # Input:
        # - im0:      original image
        # - w:        patch size (square patches of size w x w)
        # - nscales:  number of scales
        # - ngmm:     number of Gaussian components in GMM
        # [- visu     show synthesis results while estimating the model]
        # [- s        patch stride]
        # [- niter    number of iterations for estimating semi-discrete OT]
        # [- C        gradient step for estimating semi-discrete OT]
        #
        # Output:
        # - texmodel: texture model
        assert mode in ["BASETEXTO","RANDOMPATCH","NNPROJ"], "mode must be BASETEXTO, RANDOMPATCH or NNPROJ"
        
        m = im0.shape[0]
        n = im0.shape[1]
        if len(im0.shape)>2:
            nc = im0.shape[2]
        else:
            nc = 1
            
        self.nbchannels = nc
        self.nscales = nscales
        self.patchsize = w
        self.ngmm = ngmm
        self.mode = mode
        
        # Parameters
        self.s = s         # stride
        self.niter = niter  # nb of iterations in ASGD
        self.gradientstep = C          # gradient step in ASGD
        
        # Initialize lists to store transportation maps at all scales
        self.v = []
        self.y = []
        self.y2 = []
        self.nu = []
        self.gmm = []

        t0 = time.time()
              
        for scale in range(nscales-1, -1, -1):
            
            print(f'Processing scale {scale}')
            
            # Original image at current scale
            rf = 2**scale
            msc = int(np.ceil(m/rf))
            nsc = int(np.ceil(n/rf))
            im0sc = resize(im0, (msc, nsc), order=3, clip = False, anti_aliasing=False, mode='symmetric');
            im0sc2 = resize(im0, (2*msc, 2*nsc), order=3, clip = False, anti_aliasing=False, mode='symmetric');
            
            # Synthesis before transport (Gaussian or upsampled)
            if scale == nscales-1:
                (t,mv) = estimate_adsn_model(im0sc)
                self.meancolor = mv
                self.texton = t
                mv = np.reshape(mv,(1,1,nc))*np.ones((msc,nsc,nc))
                synthbt = adsn(t,mv)
            
            # Construct patch operators
            P = patch(msc,nsc,nc,w,self.s)
            if scale>0:
                P2 = patch(2*msc,2*nsc,nc,2*w,2*self.s)
                    
            # Extract patches before transport
            Pbt = P.im2patch(synthbt)
            # Source measure
            if scale == nscales-1:
                print(f'Estimate Gaussian model')
                ind = np.zeros((msc,nsc))
                ind[0:w,0:w] = 1
                meanadsnp = mv[0:w,0:w,:].flatten()
                covadsnp = get_covariance_adsn(t,ind)
                R = sqrtm(covadsnp)   # Warning: should not have complex values!
                R = np.real(R)
                sample = lambda : meanadsnp[np.newaxis,:] + (R @ np.random.randn(P.pdim,1)).T
                self.gauscov = R
                # How to initiliaze GaussianMixture with given means/covariances?
                # gmm = sklearn.mixture.GaussianMixture(reg_covar=0)
                # gmm.sample = sample
                # self.gmm.append(gmm) 
            else:
                print(f'Estimate Source GMM with {ngmm} components')
                if(self.mode=="BASETEXTO"):
                    gmm = sklearn.mixture.GaussianMixture(n_components=ngmm).fit(Pbt) # spherical or full
                    sample = lambda : gmm.sample()[0]
                elif(self.mode=="RANDOMPATCH"):
                    sample = lambda: Pbt[np.random.randint(P.Np),:]
                else:
                    sample = None
                self.gmm.append(gmm)
                
            # Target measure
            ntarget = min(P.Np, 1000)
            print(f'Estimate target measure with {ntarget} points')
            rperm = np.random.permutation(P.Np)
            P0 = P.im2patch(im0sc)
            y = P0[rperm[0:ntarget],:]
            nu = np.ones(ntarget)/ntarget
            self.y.append(y)
            self.nu.append(nu)
            if scale>0:
                P02 = P2.im2patch(im0sc2)
                y2 = P02[rperm[0:ntarget],:]
                self.y2.append(y2)
                
            # Compute semi-discrete optimal transport
            print('Compute semi-discrete optimal transport')
            if(self.mode=="NNPROJ"):
                v = 0
            else:
                v = sdot.asgd(sample,y,nu,self.niter,C)
            
            self.v.append(v)
            
            # Apply transport map to all patches
            Psynthsc,ind = sdot.map(Pbt,y,v)

            # piste 2.3 : on regarde distance de transport entre y et Psynthsc (à faire pour chauqe méthode)
            # dist(Psynthsc,y) (distance wasterstein discret discret)

            synth = P.patch2im(Psynthsc)
            
            if scale > 0: # Upsample current synthesis
                Psynth2 = y2[ind,:]
                synthbt = P2.patch2im(Psynth2)
            
            # Display
            if visu:
                dpi = 30
                fig = plt.figure(figsize=(m/float(dpi), n/float(dpi)))
                #fig = plt.subplots(1,2,constrained_layout=True)
                plt.subplot(121)
                plt.imshow(im0sc)
                plt.title('Original')
                plt.axis('off')
                plt.subplot(122)
                plt.imshow(synth)
                plt.title('Synthesis')
                fig.suptitle('Scale '+str(scale))
                plt.axis('off')
                plt.pause(0.1)
        
        elapsed_time = time.time()-t0 
        print("Elapsed time : ", elapsed_time, ' seconds')
   
    
    def synthesize(self, m, n, visu=False):
        # Texture synthesis with patch optimal transport
        #  This function allows to synthesize the texture model.
        #
        #   synth = texto_synthesize(model,M,N,use_gpu)
        #
        # Input:
        # - model: texture model
        # - m,n: desired output size
        #
        # Output:
        # - synth: synthesized texture
        
        nc = self.nbchannels
        nscales = self.nscales
        w = self.patchsize
        
        t0 = time.time()

        for scale in range(nscales-1, -1, -1):
            
            print(f'Processing scale {scale}')
            ind = nscales-1-scale
            
            # Dimensions at current scale
            rf = 2**scale
            msc = int(np.ceil(m/rf))
            nsc = int(np.ceil(n/rf))
            
            # Synthesis before transport (Gaussian or upsampled)
            if scale == nscales-1:
                mv = np.reshape(self.meancolor,(1,1,nc))*np.ones((msc,nsc,nc))
                synthbt = adsn(self.texton,mv)
                
            # Construct patch operators
            P = patch(msc,nsc,nc,w,self.s)
            if scale>0:
                P2 = patch(2*msc,2*nsc,nc,2*w,2*self.s)
                    
            # Extract patches before transport
            Pbt = P.im2patch(synthbt)
            
            # Get transportation map
            y = self.y[ind]
            v = self.v[ind]
            if scale>0:
                y2 = self.y2[ind]
            
            # Apply transport map to all patches
            Psynthsc,ind = sdot.map(Pbt,y,v)
            synth = P.patch2im(Psynthsc)
            
            if scale > 0: # Upsample current synthesis
                Psynth2 = y2[ind,:]
                synthbt = P2.patch2im(Psynth2)
            
            # Display
            if visu:
                dpi = 30
                plt.figure(figsize=(m/float(dpi), n/float(dpi)))
                plt.imshow(synth)
                plt.title('Synthesis')
                plt.axis('off')
                plt.pause(0.1)
        
        elapsed_time = time.time()-t0 
        print("Elapsed time : ", elapsed_time, ' seconds')
        
        return synth
