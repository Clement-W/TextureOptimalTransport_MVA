import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import src.texto as texto
import pickle
from os import listdir, mkdir
from os.path import isfile, isdir, join

# from gmm_ot import *
from src.patch import *
from src.gaussian_texture import *
import src.semidiscrete_ot as sdot



# Load input image
im0 = np.double(plt.imread('TextureOptimalTransport_MVA/tex/Sdesign24.png'))
m,n,nc=im0.shape

import src.texto as texto

w = 5
nscales = 4
ngmm = 4
paramstr = '_w'+str(w)+'_nscales'+str(nscales)+'_ngmm'+str(ngmm)
mode="RANDOMPATCH"
name="Sdesign24"

model = texto.model(im0, w, nscales, ngmm,mode=mode)

modelname = f"{name}_{paramstr}_{mode}.pckl"
# Save model in a file

f = open("TextureOptimalTransport_MVA/models/"+modelname, 'wb')
pickle.dump(model, f)
f.close()


synth = model.synthesize(512, 768)
synth = np.clip(synth, 0, 1)
matplotlib.image.imsave(f'TextureOptimalTransport_MVA/plots/synth_'+mode+'_w='+str(w)+'.png', synth)
