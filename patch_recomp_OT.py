import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image
import time

# from gmm_ot import *
from src.patch import *
from src.gaussian_texture import *
import src.semidiscrete_ot as sdot
import pandas as pd

# Load input image
im0 = np.double(plt.imread('TextureOptimalTransport_MVA/tex/red_peppers.png'))
m,n,nc=im0.shape

import src.texto as texto


dist_Z_R_Z_moy = []
dist_Z_R_Z_med = []
dist_Z_R_Z_weight = []
dist_X_TvX_moy = []
dist_X_TvX_med = []
dist_X_TvX_weight = []


synthesis_moy = []
synthesis_med = []
synthesis_weight = []
time_start=time.time()

#df = pd.DataFrame(columns=["dist_Z_R_Z_moy","dist_Z_R_Z_med","dist_Z_R_Z_weight","dist_X_TvX_moy","dist_X_TvX_med","dist_X_TvX_weight"])
for i in range(5):
    print('\n')
    print('-----------------------------------------')
    print('----- Processing seed no ', i,' ---------')
    print('-----------------------------------------')
    print('\n')
    np.random.seed(i)
    model_mediane = texto.model(im0, 3, 4, 4,mode="BASETEXTO",mediane=True)
    model_moy = texto.model(im0, 3, 4, 4,mode="BASETEXTO")   
    model_weight = texto.model(im0, 3, 4, 4,mode="BASETEXTO",recomp_weight=True) 
    dist_Z_R_Z_med.append(model_mediane.dist_Z_R_Z)
    dist_Z_R_Z_moy.append(model_moy.dist_Z_R_Z)
    dist_Z_R_Z_weight.append(model_weight.dist_Z_R_Z)

    dist_X_TvX_med.append(model_mediane.dist_X_TvX)
    dist_X_TvX_moy.append(model_moy.dist_X_TvX)
    dist_X_TvX_weight.append(model_weight.dist_X_TvX)
    print("Time elapsed after seed ",i," : ",(time.time()-time_start)/60," minutes")

dist_Z_R_Z_moy = np.array(dist_Z_R_Z_moy)
dist_Z_R_Z_med = np.array(dist_Z_R_Z_med)
dist_Z_R_Z_weight = np.array(dist_Z_R_Z_weight)
dist_X_TvX_moy = np.array(dist_X_TvX_moy)
dist_X_TvX_med = np.array(dist_X_TvX_med)
dist_X_TvX_weight = np.array(dist_X_TvX_weight)

"""
# average over the 5 runs
avg_dist_Z_R_Z_moy = np.mean(dist_Z_R_Z_moy,axis=0)
avg_dist_Z_R_Z_med = np.mean(dist_Z_R_Z_med,axis=0)
avg_dist_Z_R_Z_weight = np.mean(dist_Z_R_Z_weight,axis=0)
avg_dist_X_TvX_moy = np.mean(dist_X_TvX_moy,axis=0)
avg_dist_X_TvX_med = np.mean(dist_X_TvX_med,axis=0)
avg_dist_X_TvX_weight = np.mean(dist_X_TvX_weight,axis=0)

std_dist_Z_R_Z_moy = np.std(dist_Z_R_Z_moy,axis=0)
std_dist_Z_R_Z_med = np.std(dist_Z_R_Z_med,axis=0)
std_dist_Z_R_Z_weight = np.std(dist_Z_R_Z_weight,axis=0)
std_dist_X_TvX_moy = np.std(dist_X_TvX_moy,axis=0)
std_dist_X_TvX_med = np.std(dist_X_TvX_med,axis=0)
std_dist_X_TvX_weight = np.std(dist_X_TvX_weight,axis=0)
"""

# save all the results
np.savez("patchrecomp.npz",dist_Z_R_Z_moy=dist_Z_R_Z_moy,dist_Z_R_Z_med=dist_Z_R_Z_med,
         dist_Z_R_Z_weight=dist_Z_R_Z_weight,dist_X_TvX_moy=dist_X_TvX_moy,
         dist_X_TvX_med=dist_X_TvX_med,dist_X_TvX_weight=dist_X_TvX_weight)
