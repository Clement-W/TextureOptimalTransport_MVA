import numpy as np
import ot

# Example data, replace this with your actual data
num_patches_distribution1 = 1000
num_patches_distribution2 = 2112
patch_dimension = 27

# Generate random patches for demonstration
patches_distribution1 = np.random.rand(num_patches_distribution1, patch_dimension)
patches_distribution2 = np.random.rand(num_patches_distribution2, patch_dimension)

# Assign uniform weights to patches (can be replaced with actual weights)
weights_distribution1 = np.ones(num_patches_distribution1) / num_patches_distribution1
weights_distribution2 = np.ones(num_patches_distribution2) / num_patches_distribution2

# Calculate the Euclidean distance matrix between patches
distance_matrix = ot.dist(patches_distribution1, patches_distribution2)
distance_matrix /= distance_matrix.max()  # Normalize the distance matrix

# Calculate Wasserstein distance using ot.emd2
wasserstein_distance = ot.emd2(weights_distribution1, weights_distribution2, distance_matrix)

print(f"Wasserstein Distance: {wasserstein_distance}")
