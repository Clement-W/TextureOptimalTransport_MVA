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


mode="BASETEXTO"
texture="ground1013_small"

# Load input image
im0 = np.double(plt.imread('TextureOptimalTransport_MVA/tex/Sdesign24.png'))
m,n,nc=im0.shape

for w in [3]:
    print('\n')
    print('-----------------------------------------')
    print('----- Processing texture ', texture,' -------')
    print('-----------------------------------------')
    print('\n')

    model = texto.model(im0, w, 4, 4,mode=mode)
    wasserstein_distance = model.wasserstein
    print(f"Wasserstein Distances: {wasserstein_distance}")
    np.save(f'wasserstein/wasserstein_{mode}_{texture}_{w}.npy', wasserstein_distance)
    #synth = model.synthesize(512, 768)
    #synth = np.clip(synth, 0, 1)
    #matplotlib.image.imsave(f'synth_GMM_NNProj_Sdesign.png', synth)
