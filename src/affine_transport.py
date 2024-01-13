import numpy as np
from scipy.linalg import sqrtm

def affine_transport(source_dist, target_dist):
    """
    Calculate the affine transport map between two Gaussian distributions.
    
    :param source_mean: Mean of the source Gaussian distribution.
    :param source_cov: Covariance matrix of the source Gaussian distribution.
    :param target_mean: Mean of the target Gaussian distribution.
    :param target_cov: Covariance matrix of the target Gaussian distribution.
    :return: The transformation matrix A and the offset vector b.
    """
    
    source_mean = np.mean(source_dist, axis=0)
    source_cov = np.cov(source_dist, rowvar=False)
    target_mean = np.mean(target_dist, axis=0)
    target_cov = np.cov(target_dist, rowvar=False)

    # Compute the square root of the target covariance matrix
    sqrt_target_cov = np.real(sqrtm(target_cov))

    # Compute the transformation matrix A and the offset vector b
    A = sqrt_target_cov @ np.real(sqrtm(np.linalg.inv(source_cov)))
    b = target_mean - A @ source_mean

    return np.real(A),np.real(b)


def create_affine_transform(A, b):
    def affine_transform(x):
        return A @ x + b
    return affine_transform

