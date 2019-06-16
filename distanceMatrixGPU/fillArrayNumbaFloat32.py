from numba import cuda, uint16, float32
cuda_type = float32
@cuda.jit(argtypes=[cuda_type[:], cuda_type[:,:],uint16])
def fill_FlattenTraingle(dist_out, X, n):
    """Fill a sub-triangle under and over the diagonal of array with distance data.

     Parameters
    ----------
    dist_out: array
        The part of the array that will be used in the complete array.
    X: array
        Look-up table the with the distances.
    n: int
        Length of the the trigle-matrx.
            
    Returns
    -------
    Void
    """
    x, y = cuda.grid(2)
    dx, dy = cuda.gridsize(2) 
    if x >= X.shape[0] and y >= X.shape[1]:
        return
    
    for i in range(x, n , dx):
        for k in range(y, i, dy):
            ix = k*n - k*(k +1)/2 + (i - 1 - k)
            p = X[int(i), int(k)]
            dist_out[int(ix)] = p

def cudaFillFlattenArray32(x, X, N):
    """Fill flatten sub-array with gpu.

    Parameters
    ----------
    x: array
        The array to fill with distance data.
    X: array
        Look-up table the with the distances.
    N: int
        Length of the the trigle-matrx.
            
    Returns
    -------
    Void
    """
    dist_out = cuda.to_device(x)
    X = cuda.to_device(X)
    threadsPerBlock = 256
    numBlocks = int((X.shape[0] + threadsPerBlock - 1)/threadsPerBlock)
    fill_FlattenTraingle[threadsPerBlock,numBlocks](dist_out,X,N)
    
    return dist_out.copy_to_host()

@cuda.jit(argtypes=[cuda_type[:], cuda_type[:,:],uint16])
def fillFull_traingle(dist_out, subarr, n):
    """Fill a sub-triangle under and over the diagonal of array with distance data.

    Parameters
    ----------
    dist_out: array
        The part of the array that will be used in the complete array.
    subarr: array
        Look-up table the with the distances.
    n: int
        Length of the the trigle-matrx.
            
    Returns
    -------
    Void
    """
    x, y = cuda.grid(2)
    dx, dy = cuda.gridsize(2) 
    if x >= subarr.shape[0] and y >= subarr.shape[1]:
        return
    for i in range(x, n , dx):
        for k in range(y, i , dy):
            ix = k*n - k*(k +1)/2 + (i - 1 - k)
            subarr[int(i), int(k)] = dist_out[int(ix)]
            subarr[int(k), int(i)] = dist_out[int(ix)]
        
def cudaFillFullArray32(x, X, N):
    """Fill Full sub-array with gpu.

    Parameters
    ----------
    x: array
        The array to fill with distance data.
    X: array
        Look-up table the with the distances.
    N: int
        Length of the the trigle-matrx.
            
    Returns
    -------
    Void
    """
    dist_out = cuda.to_device(x)
    subarr = cuda.to_device(X)
    threadsPerBlock = 256
    numBlocks = int((X.shape[0] + threadsPerBlock - 1)/threadsPerBlock)
    fillFull_traingle[threadsPerBlock, numBlocks](dist_out, subarr, N)
    return subarr.copy_to_host()

