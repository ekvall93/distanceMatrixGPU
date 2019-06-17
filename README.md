# DistanceMatrixGPU
GPU accelerated distance matrix calculation

## Getting Started

### Prerequisites

python3.6 has been used to test out the repository. The installation uses anaconda, but it is not necessary.

A GPU that uses CUDA is necessary.


### Installing

Use this following command to install all required packages.

```
pip install git+https://github.com/ekvall93/distanceMatrixGPU.git
```

You need to set correct path to cuda

```
export PATH=/usr/local/cuda-X.X/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-X.X/lib64\{LD_LIBRARY_PATH:+:${$LIBRARY_PATH}}
```

Here X.X is you cuda version e.g., 9.2, 10.0 etc. The code have only been tested for cuda9.2 and cuda10.0. 


## Running the tests

Initialization
```
from numbaDistanceMatrix.cudaDistanceMatrix import DistanceMatrix
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
n = 200
X = np.random.rand(n, n).astype(np.float32)
```
Calculate the distance matrix

```
DM = DistanceMatrix()
DM.calculate_distmatrix(X)
```

Get particular entry in distance matrix.

```
DM.get_similarity(10,2)
#0.77965623
cosine_similarity(X)[10,2]
#0.77965623
```

Check correctness for under triangle.
```
SKlearn_under = cosine_similarity(X)[np.tril_indices(n, k=-1)]
under_dist = DM.get_distance_matrix(fullMatrix=False)
np.allclose(np.sort(under_dist), np.sort(SKlearn_under))
#True
```
Retrieve full distance matrix if necessary.
```
SKlearn_full = cosine_similarity(X)
DM_full = DM.get_distance_matrix(fullMatrix=True)
np.allclose(SKlearn_full, DM_full)
#True
```



## Run time performance increase.

The GPU allows for substantial speed up for larger matrices. It speeds up both thicker and smaller matrices (see figure one below).


![alt text](/images/GPUNN.png)

![alt text](/images/GPUNN10.png)


## Memory efficency

Efficient memory usage is attained by only calculating the under-triangle. This is possible since the distance matrix is symmetric i.e., D(i,j) = D(j,i). Only calculating the under-triangle is done by dividing the matrix into smaller sub-matrices and only storing the values below the diagonal, see figure two below. 


![alt text](/images/subDot.png)

The memory storage is

<a href="https://www.codecogs.com/eqnedit.php?latex=\frac{n^{2}&space;-&space;n}{2}=O(n^{2})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{n^{2}&space;-&space;n}{2}=O(n^{2})" title="\frac{n^{2} - n}{2}=O(n^{2})" /></a>

Even though it is not asymptotically better, it still used less than half of the memory.

To find the entry for (i,j) still only takes

<a href="https://www.codecogs.com/eqnedit.php?latex=O(1)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?O(1)" title="O(1)" /></a>

Since there is a unique mapping for each tuple (i,j) to l in the under triangle, see figure below.
![alt text](/images/mapping.png)

For completeness, if j=1,2,3,...,(n-1) and i < j then just use equation 1 in the figure above. However, use i -> j and j -> i since it's the matrix D is symmetrical i.e., D(i,j)=D(j,i)=A(l(i,j))= A(l(j,i)). For instance, D(2,3)=D(3,2)=A(l(2,3))=A(l(3,2)). In the third case when i=j, the output is always 1 since a distance matrix D(i,j)=D(i,i)=1. Hence, the diagonal dont need to be explicity stored.

## Authors

* **Markus Ekvall**

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details




