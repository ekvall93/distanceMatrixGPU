# DistanceMatrixGPU
GPU accelerated distance matrix calculation

##Runt time peforamnce increse.

The GPU allows for substantial speed up for larger matrices. Furthermore, the computational speedup is obtained for both thicker and smaller matrices, see figures below.


![alt text](/images/GPUNN.png)

![alt text](/images/GPUNN10.png)


##Memory efficency

To be more memory efficient only the under triangle is calculated. This is done by dividing the matrix into smaller sub-matrices and only storing the values below the diagonal, see figure below. This is possible since the distance matrix is symmetric i.e., D(i,j) = D(j,i). 


![alt text](/images/subDot.png)

The memory storage is still $O(n^{2})$