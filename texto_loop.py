"""
 Texture synthesis with patch optimal transport

 Execute this script to process all textures placed in the folder "tex/"

 Copyright Arthur Leclaire (c), 2019.

"""

import numpy as np
import matplotlib.pyplot as plt
import src.texto as texto
import pickle
import time
import os
import pandas as pd
from os import listdir, mkdir
from os.path import isfile, isdir, join

os.chdir('TextureOptimalTransport_MVA')

if not isdir('synth'):
    mkdir('synth')
if not isdir('models'):
    mkdir('models')
if not isdir('wasserstein'):
    mkdir('wasserstein')

files = [f for f in listdir('tex') if isfile(join('tex', f))]
files.sort()

plt.ion()



for z in range(10):
    print("Seed: ",z)
    np.random.seed(z)
    results_ot=pd.DataFrame(columns=["BASETEXTO","RANDOMPATCH","NNPROJ","AFFINE"])
    results_time=pd.DataFrame(columns=["BASETEXTO","RANDOMPATCH","NNPROJ","AFFINE"])
    if os.path.exists(f"results_ot_{z}.csv"):
        results_ot=pd.read_csv(f"results_ot_{z}.csv")
    if os.path.exists(f"results_time_{z}.csv"):
        results_time=pd.read_csv(f"results_time_{z}.csv")
    for ind in range(len(files)):
        print(ind)
        name, ext = files[ind].rsplit('.', 1)

        print('\n')
        print('-----------------------------------------')
        print('----- Processing texture no ', ind, ' ', name,' -------')
        print('-----------------------------------------')
        print('\n')

        # Load input image
        im0 = np.double(plt.imread('tex/'+name+'.'+ext))
        if im0.ndim < 3:
            im0 = im0[:, :, np.newaxis]
        m, n, nc = im0.shape

        # Parameters
        ws = [3]
        nscales = 4
        ngmm = 4

        
        results_ot.loc[name,:]=np.nan
        results_time.loc[name,:]=np.nan
        
        
        for w in ws:    
            paramstr = '_w'+str(w)+'_nscales'+str(nscales)+'_ngmm'+str(ngmm)
            doestimation = True

            for mode in ["BASETEXTO","RANDOMPATCH","NNPROJ","AFFINE"]:
                print("Mode: ", mode,"w :",w)
                modelname = f"{name}_{paramstr}_{mode}.pckl"
                file_path = "wasserstein/" + modelname
                doestimation = results_ot.loc[name,mode] is np.nan
                print("doestimation: ",doestimation)

                if not doestimation:
                    time1 = time.time()
                    # Model estimation
                    model = texto.model(im0, w, nscales, ngmm,mode=mode)

                    # Save model in a file
                    results_ot.loc[name,mode]=model.wasserstein
                    results_time.loc[name,mode]=time.time()-time1
                    
                    results_ot.to_csv(f"results_ot_{z}.csv")
                    results_time.to_csv(f"results_time_{z}.csv")

                    #f = open("wasserstein/"+modelname, 'wb')
                    #pickle.dump(model.wasserstein, f)
                    #f.close()
                    
                    # Save model in a file
                    f = open("models/"+modelname, 'wb')
                    pickle.dump(model, f)
                    f.close()
                """
                # Load texture model from pre-computed file
                f = open("models/"+modelname, 'rb')
                model = pickle.load(f)
                f.close()
                    
                # Synthesis
                M, N = 512, 768
                synth = model.synthesize(M, N)

                # Save Results
                if nc == 1:
                    synth = synth[:, :, 0]
                    im0 = im0[:, :, 0]

                plt.imsave('synth/'+name+'_synth'+paramstr+'_'+mode+'.'+ext, synth, cmap='Greys')
                plt.imsave('synth/'+name+'_'+mode+'.'+ext, im0, cmap='Greys')
                """
