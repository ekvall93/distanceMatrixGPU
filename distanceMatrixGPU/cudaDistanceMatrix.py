import h5py
import numpy as np
from skcuda import linalg, misc
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
linalg.init()
from numpy.linalg import norm
import os.path

from .fillArrayNumbaFloat32 import cudaFillFlattenArray32, cudaFillFullArray32

class DistanceMatrix:
    """GPU accelerated Distannce matrix caluclations.
    Parameters
    ----------
    file_name: string
        Name of h5py file to save the distance matrix (don't need to have whole array in working memory).
    dtype_out: string
        What precision is needed? 32 or 64?
    gpu: bool
        Use gpu accelerations
    numba: bool
        Use additional gpu acceleration

    """
    def __init__(self,
            file_name="cuda_dist",
            gpu=True,
            numba=True,
            dictinoray=None):
        if os.path.isfile(str(file_name)):
            os.remove(str(file_name))
        
        self.filename_dist = file_name
        self.file_dist = h5py.File(file_name, 'w')
        self.out_type = np.float32
        self.float = "float32"
        self.load_full = False
        self.gpu = gpu
        self.numba = numba

        if dictinoray:
            self.dictinoray = dictinoray
        else:
            self.dictinoray = None
    def _cuda_norm(self, X):
        """Caluclate L2-norm on gpu.

        Parameters
        ----------
        X: array
            Array to normalize
        Returns
        -------
        normX: array
            Normalized array

        """
        return misc.divide(X, misc.sum(X ** 2, axis=1, keepdims=True) ** 0.5)    
    def _norm(self, X):
        """Caluclate L2-norm on cpu.

        Parameters
        ----------
        X: array
            Array to normalize
        Returns
        -------
        normX: array
            Normalized array
            
        """
        return X / norm(X,axis=1, keepdims=True)

        
    def _get_XTX_cuda(self, X, x_1, x_2, y_1, y_2):
        """Caluclate dot product between two array on gpu.

        Parameters
        ----------
        X: array
            Array to normalize
        x_1: int
            Lower bound on slice on x-axis
        x_2: int
            Upper bound on slice x-axis
        y_1: int
            Lower bound on slice y-axis
        y_2: int
            Upper bound on slice y-axis
        Returns
        -------
        XX.T: array
            X X.T array
            
        """
        X_f, X_b = gpuarray.to_gpu(X[x_1:x_2, :]), gpuarray.to_gpu(X[y_1:y_2, :])
        X_f_norm, X_b_norm = self._cuda_norm(X_f), self._cuda_norm(X_b)
        return linalg.dot(X_f_norm, X_b_norm, transb="T").get()
    
    def _get_XTX(self, X, x_1, x_2, y_1, y_2):
        """Caluclate dot product between two array on cpu.

        Parameters
        ----------
        X: array
            Array to normalize
        x_1: int
            Lower bound on slice on x-axis
        x_2: int
            Upper bound on slice x-axis
        y_1: int
            Lower bound on slice y-axis
        y_2: int
            Upper bound on slice y-axis
        Returns
        -------
        XX.T: array
            X X.T array
            
        """
        Xnorm1, Xnorm2 = self._norm(X[x_1:x_2,:]), self._norm(X[y_1:y_2,:])
        return np.dot(Xnorm1, Xnorm2.T)

    def _get_array_extensions(self, i, j):
        """Get how much sub-array have to be exteneded to completely fill the whole array.

        Parameters
        ----------
        i: int
            Index to keep track on size of and position on sub-array
            
        j: int
            Index to keep track on size of and position on sub-array
            
        Returns
        -------
        extendX: int
            How much to extend sub-array on x-axis
        extendY: int
            How much to extend sub-array on y-axis
        """
        extendX, extendY = 0, 0
        if i == self.nr_square - 1 and j == self.nr_square - 1:
            extendX = self.N % self.nr_square
            extendY = self.N % self.nr_square
        elif i == self.nr_square - 1:
            extendX = self.N % self.nr_square 
        elif j == self.nr_square - 1:
            extendY = self.N % self.nr_square 
        return extendX, extendY
    
    def _get_extensions_coef(self, i, j, dx, dy):
        """Get position of where sub-array will be spliced from the complete array.

        Parameters
        ----------
        i: int
            Index to keep track on size of and position on sub-array
            
        j: int
            Index to keep track on size of and position on sub-array
        dx: int
            How much to extend on x-axis
        dy: int
            How much to extend on y-axis
            
        Returns
        -------
        x_1: int
            Lower bound on slice on x-axis
        x_2: int
            Upper bound on slice on x-axis
        y_1: int
            Lower bound on slice on y-axis
        y_2: int
            Upper bound on slice on y-axis
        """
        x_1, x_2 = self.l * i, self.l * (i + 1) + dx
        y_1, y_2 = self.l * j, self.l * (j + 1) + dy
        return x_1, x_2, y_1, y_2
    
    def _get_flatten_distance_matrix(self, f_dist):
        """Get the complete flatten distance matrix.

        Parameters
        ----------
        f_dist: array
            Contrains all sub-arrays with all distance data.
       
            
        Returns
        -------
        flattenDistancematrix: array
            The whole distance matrix
        """
        entries = int(self.N * (self.N - 1) / 2)
        distance_matrix = np.zeros([entries],dtype=self.out_type)
        start_splice =  0
        for k in list(f_dist.keys()):
            data = np.asarray(f_dist[k])
            end_splice = start_splice + data.shape[0]
            distance_matrix[start_splice:end_splice] = data
            start_splice = end_splice
        return distance_matrix

    def _fill_fullDistanceMatrix(self, i, j, dist_data):
        """Fill the complete full distance matrix (Void, fill array in-place)

        Parameters
        ----------
        i: int
            Index to keep track on size of and position on sub-array
            
        j: int
            Index to keep track on size of and position on sub-array
        dist_data: array
            Flatten sub-array with the distance data.
            
        Returns
        -------
        Void
        
        """
        extendX, extendY = self._get_array_extensions(i, j)
        x_1, x_2, y_1, y_2 = self._get_extensions_coef(i, j, extendX, extendY)    
        if i == j:
            subArr = np.ones([self.l + extendX, self.l + extendX], dtype=self.out_type)
            if self.float == "float32":
                A = cudaFillFullArray32(dist_data, subArr, self.l + extendX)
            elif self.float == "float64":
                A = cudaFillFullArray64(dist_data, subArr, self.l + extendX)
            self.fullDistMatrix[x_1: x_2, y_1 : y_2] = A    
        else:
            self.fullDistMatrix[x_1: x_2, y_1 : y_2] = dist_data.reshape(self.l + extendX, self.l+ extendY)
            self.fullDistMatrix[y_1 : y_2, x_1: x_2] = dist_data.reshape(self.l + extendX, self.l+ extendY).T

    
    def _get_full_distance_matrix(self, f_dist):
        """Get the complete full distance matrix

        Parameters
        ----------
        f_dist: array
            Array with all sub-arrays with distance data.
        Returns
        -------
        fullDistanceMatrix: array
            Array with the full complete distance array
        """
        self.fullDistMatrix = np.zeros([self.N, self.N])
        sorted_filenames = self._sort_files(f_dist)
        for fileArr in sorted_filenames:
            dist_data = np.asarray(f_dist[fileArr])
            i, j = int(fileArr.split(":")[1].split("_")[1]), int(fileArr.split(":")[1].split("_")[2])
            self._fill_fullDistanceMatrix(i, j, dist_data)
        return self.fullDistMatrix

    def _sort_files(self, f_dist):
        file_names = list(f_dist.keys())
        arr_number = [int(ss.split(":")[1].split("_")[3]) for ss in file_names]
        _, sorted_filenames = zip(*sorted(zip(arr_number, file_names)))
        return sorted_filenames

 
    def get_distance_matrix(self, fullMatrix=False):
        """Get either the complete flatten or full distance matrix.

        Parameters
        ----------
        fullMatrix: bool
            Get full (True) take N**2 memory, else flatten (False) take N(N-1)/2 memory. Both O(N**2)
        Returns
        -------
        DistanceMatrix: array
            Matrix with all the distance.
        """
        f_dist = h5py.File(self.filename_dist, 'r')
        if fullMatrix:
            self.load_full = fullMatrix
            return self._get_full_distance_matrix(f_dist)
        else:
            return self._get_flatten_distance_matrix(f_dist)
    
    def _get_flatten_entries(self, i, j, X_shape):
        """Get entries for the flatten array. It's a bi-jection (i,j) -> k, k -> (i,j)

        Parameters
        ----------
        i : int
            Index for the x-axis i.e., the ith row.     
        j: int
            Index for the y-axis i.e., the jth column.     
        X_shape: tuple
            Shape of sub-array
        Returns
        -------
        l_x: int
            length of sub-array on x-axis
        entries: int
            Number of entries in flatten array
        square: bool
            Is the matrix a square(True) or a triangle(False).
        """
        if i == j:
            l_x =  X_shape[0]
            entries = l_x*(l_x - 1)/2
            square = False
        else:
            l_x, l_y = X_shape[0], X_shape[1]                
            entries = l_x * l_y
            square = True
        return l_x, entries, square
    
    def _fill_flatten_distMatrix(self, entries, X, dx):
        """Fill the flatten distance matrix with data from sub-arrays.

        Parameters
        ----------
        entries : int
            Number of entries in flatten array
        X: array
            Sub-array with distance data.
        dx: int
            Length of the sub-array of the complete array to get filled.
        Returns
        -------
        subdistMatrix_flatten: array
            Part of the complete flatten array that have been given distance values. 
        """
        empty_subdistMatrix_flatten = np.zeros((int(entries)), dtype=self.out_type)
        if self.float == "float32":
            subdistMatrix_flatten = cudaFillFlattenArray32(empty_subdistMatrix_flatten, X, dx)
        elif self.float == "float64":
            subdistMatrix_flatten = cudaFillFlattenArray64(empty_subdistMatrix_flatten, X, dx)
        return subdistMatrix_flatten
    
    def get_similarity(self, i, j,load_full=False):
        """Fill the flatten distance matrix with data from sub-arrays.

        Parameters
        ----------
        entries : int
            Number of entries in flatten array
        X: array
            Sub-array with distance data.
        dx: int
            Length of the sub-array of the complete array to get filled.
        Returns
        -------
        subdistMatrix_flatten: array
            Part of the complete flatten array that have been given distance values. 
        """
        if isinstance(i, str) or isinstance(j, str):
            if self.dictinoray:
                try:
                    i = int(self.dictinoray[i])
                    j = int(self.dictinoray[j])
                except:
                    print("The values dont exists in dict.")
        
        if not self.load_full:
            if load_full:
                self.load_full = load_full
                f_dist = h5py.File(self.filename_dist, 'r')
                self._get_full_distance_matrix(f_dist)
                return self.fullDistMatrix[i, j]
            else:
                if i == j:
                    return 1
                f_dist = h5py.File(self.filename_dist, 'r')
                sorted_filenames = self._sort_files(f_dist)
                if i < j:
                    f_n = self._get_val(sorted_filenames, j, i)
                    val = self._get_ix(f_dist,f_n,j, i)
                
                else:   
                    f_n = self._get_val(sorted_filenames, i, j)
                    val = self._get_ix(f_dist,f_n,i,j)
                return val
        else:
            return self.fullDistMatrix[i, j]
    
    def most_similar(self, i, load_full=False):
        """Fill the flatten distance matrix with data from sub-arrays.

        Parameters
        ----------
        entries : int
            Number of entries in flatten array
        X: array
            Sub-array with distance data.
        dx: int
            Length of the sub-array of the complete array to get filled.
        Returns
        -------
        subdistMatrix_flatten: array
            Part of the complete flatten array that have been given distance values. 
        """
        if isinstance(i, str) or isinstance(j, str):
            if self.dictinoray:
                try:
                    i = int(self.dictinoray[i])
                except:
                    print("The values dont exists in dict.")
        


        if not self.load_full:
            if load_full:
                self.load_full = load_full
                f_dist = h5py.File(self.filename_dist, 'r')
                self._get_full_distance_matrix(f_dist)

                sims = self.fullDistMatrix[i, :]
            else:
                sims = list()
                f_dist = h5py.File(self.filename_dist, 'r')
                sorted_filenames = self._sort_files(f_dist)
                
                for j in range(self.N):
                    if i == j:
                        val = 1
                    elif i < j:
                        f_n = self._get_val(sorted_filenames, j, i)
                        val = self._get_ix(f_dist,f_n,j, i)
                    else:
                        f_n = self._get_val(sorted_filenames, i, j)
                        val = self._get_ix(f_dist,f_n,i,j)
                    sims.append(val)
                sims = np.asarray(sims)

        else:
            sims = self.fullDistMatrix[i, :]

        if self.dictinoray:
            sorted_val, sorted_name = zip(*sorted(zip(sims, self.dictinoray.keys())))
            return dict(zip(sorted_name[::-1], sorted_val[::-1]))
        else:
            return self.fullDistMatrix[i, :]
                
    
    def _get_bounderies(self, f_n):
        """Get bounderies of sub-array in the array.

        Parameters
        ----------
        f_n : string
            File name of sub-array
        Returns
        -------
        x_range[0]: int
            Start cordinate in x-axis
        x_range[1]: int
            Stop cordinate in x-axis
        y_range[0]: int
            Start cordinate in y-axis
        y_range[1]: int
            Stop cordinate in y-axis
        """
        x, y = f_n.split(":")[0].split("_")
        x_range = x.split("-")
        y_range = y.split("-")
        return x_range[0], x_range[1], y_range[0], y_range[1]
    
    def _get_val(self, x, i, j):
        """Get (i,j) from under triangle.

        Parameters
        ----------
        x : array
            Data array
        i: int
            x index
        j: int
            y index
        Returns
        -------
        val: int
            (i,j) value
        """
        if len(x) == 1:
            f_n = x[0]
            x1, x2, y1, y2 = self._get_bounderies(f_n)
            if int(x1) <= i <= int(x2) and int(y1) <= j <= int(y2):
                return f_n
            else:
                return None
        else:
            p = int(np.ceil(len(x) / 2))
            x1, x2, y1, y2 = self._get_bounderies(x[p])
         
            if int(x1) <= i:
                if i <= int(x2):
                    if int(y1) <= j:
                        if j >= int(y2):
                            val = self._get_val(x[p], i, j)
                        else:
                            val = self._get_val(x[p:], i, j)
                    else:
                        val = self._get_val(x[:p], i, j)
                else:
                    val = self._get_val(x[p:], i, j)
            else:
                val = self._get_val(x[:p], i, j)
      
        return val
    
    def _get_ix(self, f_dist, file_name, i, j):
        """Get (i,j) from under triangle.

        Parameters
        ----------
        f_dist : object
            Sotre all file data
        file_name: file_name
            Names of subarrays
        i: int
            x index
        j: int
            y index
        Returns
        -------
        val: int
            (i,j) value
        """
        f1, f2 = file_name.split(":")
        val = f2.split("_")
        dx, dy = f1.split("_")
        x1, x2 = dx.split("-")
        y1, y2 = dy.split("-")
        
        si, sj = int(val[1]), int(val[2])
        X = np.array(f_dist[file_name])
        
        di, dj = int(i) - int(x1), int(j) - int(y1)
        extendX, extendY = self._get_array_extensions(si, sj)
        if si == sj:
            if self.float == "float32":
                subArr = np.ones([self.l + extendX, self.l + extendX], dtype=self.out_type)
                A = cudaFillFullArray32(X, subArr, self.l + extendX) 
        else:
            A = X.reshape(self.l + extendX, self.l+ extendY)
        
        return A[di, dj]


    
    def calculate_distmatrix(self, X, nr_square=4):
        """Calcualte the whole distane matrix

        Parameters
        ----------
        X : array
            Array to calcualte distances for.
        Returns
        -------
        Void
        """
        if np.float32 != X.dtype:
            print("Warning: Array is not float32. The array will be convered from ", X.dtype, " to float32 for speed up.")
            X = X.astype(np.float32)
        l = X.shape[0] / nr_square
        assert l >= 2, "Please pick fewer number of sub-arrays each has a length longer than 2 elements. Currently " + str(round(l, 2))
        if self.float=="float64":
            assert X.shape[0] >= 10000, "If using float64 ensure that that X.shape[0] > 10 000. Otherwise use float32 or cpu-version."
        self.l = int(np.floor(l))
        self.N = X.shape[0]
        self.nr_square = nr_square
        arr_nr = 0
        for i in range(self.nr_square):
            for j in range(self.nr_square):
                
                if j <= i:
                    extendX, extendY = self._get_array_extensions(i, j)
                    x_1, x_2, y_1, y_2 = self._get_extensions_coef(i, j, extendX, extendY)
                    
                    if self.gpu:
                        XTX = self._get_XTX_cuda(X,*[x_1,x_2,y_1,y_2])
                    else:
                        XTX = self._get_XTX(X, *[x_1, x_2, y_1, y_2])
                        
                    l_x, entries, square = self._get_flatten_entries(i, j, XTX.shape)
                    if self.numba:       
                        if square:
                            subdistMatrix_flatten = XTX.flatten()
                        else:
                            subdistMatrix_flatten = self._fill_flatten_distMatrix(entries, XTX, l_x)
                
                    else:
                        if square:
                            subdistMatrix_flatten = XTX.flatten()
                        else:
                            subdistMatrix_flatten = XTX[np.tril_indices(l_x, k=-1)]
                    
                    self.file_dist.create_dataset(str(x_1) + "-" + str(x_2 - 1) + "_" + str(y_1)+"-"+ str(y_2 - 1) + ':subArray_' + str(i) + "_" + str(j) + "_" + str(arr_nr), data=subdistMatrix_flatten, dtype=subdistMatrix_flatten.dtype)
                    arr_nr +=1
                
        self.file_dist.close()
