import ot
import numpy as np

def wasserstein_distance(a,b):
    sum_a = a.sum(0)
    sum_b = b.sum(0)
    norm_a = a/sum_a
    norm_b = b/sum_b
    M=ot.dist(norm_a,norm_b)
    M /= M.max()

    shp_a=norm_a.shape[0]
    shp_b=norm_b.shape[0]

    weights_a=np.ones(shp_a)/shp_a
    weights_b=np.ones(shp_b)/shp_b

    return ot.emd2(weights_a, weights_b, M,numItermax=500000)