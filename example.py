from cudaDistanceMatrix import DistanceMatrix
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

#It is possible to retrieve the full distance matrix if necessary.
n = 200
X = np.random.rand(n, n).astype(np.float32)

#Initialize DistanceMatrix object and calculate the distance matrix.

DM = DistanceMatrix()
DM.calculate_distmatrix(X)

#Get specific value in the distance matrix.

print("Sklearn: entry (10,2)")
print(DM.get_similarity(10, 2))

print("CudaDM: entry (10,2)")
print(cosine_similarity(X)[10, 2])

#Retrieve the flatten (under-triangle) distance matrix and compare it to Sklearn's version.

SKlearn_under = cosine_similarity(X)[np.tril_indices(n, k=-1)]
under_dist = DM.get_distance_matrix(fullMatrix=False)
print(np.allclose(np.sort(under_dist), np.sort(SKlearn_under)))

#It is possible to retrieve the full distance matrix if necessary.

SKlearn_full = cosine_similarity(X)
DM_full = DM.get_distance_matrix(fullMatrix=True)

print(np.allclose(SKlearn_full, DM_full))
