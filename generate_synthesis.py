import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from src.patch import *
#list files in models folder
files = os.listdir('TextureOptimalTransport_MVA/models')


for file in files:
    if(file.endswith('.pckl')):
        

        
        print(file)
        f = open('TextureOptimalTransport_MVA/models/' + file, 'rb')
        # get name of the imgae from file
        image_name = file.split('_')[0] + '.png'
        file_path = f'TextureOptimalTransport_MVA/tex/{image_name}'
        if os.path.exists(file_path):
            print(image_name)
            # load image to get its dim
            im0 = np.double(plt.imread())
            m,n,nc=im0.shape
            model = pickle.load(f)
            f.close()

            # extract last synthesis
            last_synthesis_patches = model.patches_after_transport[-1]
            w=int(file[6])
            P = patch(m,n,nc,w,1)
            synth = P.patch2im(last_synthesis_patches)
            name=file.split('.')[0]
            plt.imsave(f'TextureOptimalTransport_MVA/synth/{name}_synth.png', np.clip(synth,0,1))