import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image

# from gmm_ot import *
from patch import *
from gaussian_texture import *
import semidiscrete_ot as sdot


# Load input image
im0 = np.double(plt.imread('tex/Sdesign24.png'))
m,n,nc=im0.shape

import texto

model = texto.model(im0, 3, 4, 4,mode="BASETEXTO")   
synth = model.synthesize(512, 768)
synth = np.clip(synth, 0, 1)
matplotlib.image.imsave(f'synth_GMM_NNProj.png', synth)
