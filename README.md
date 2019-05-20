# DistanceMatrixGPU
GPU accelerated distance matrix calculation

## Getting Started

### Prerequisites

python3.6 has been used to test out the repository. he installation uses anaconda, but it is not necessary.

A GPU that uses CUDA is necessary.


### Installing

Use this following command to install all required packages.

```
bash install.sh
```

## Runt time peforamnce increse.

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



