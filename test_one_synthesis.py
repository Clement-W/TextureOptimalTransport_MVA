import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import src.texto as texto


# Load input image
im0 = np.double(plt.imread('tex/ground1013_small.png'))
m,n,nc=im0.shape

model = texto.model(im0, 3, 4, 4,mode="BASETEXTO")   
synth = model.synthesize(512, 512)
synth = np.clip(synth, 0, 1)
matplotlib.image.imsave(f'synth.png', synth)
