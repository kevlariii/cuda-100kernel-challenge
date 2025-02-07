from numba import cuda
import numpy as np
from math import ceil
import numba


@cuda.jit
def scalar_product(x, y, result, blocksize):

    idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if idx < len(x):
        temp = x[idx] * y[idx]
    else: temp = 0
    
    #assign a shared memory within each block for faster access
    shared = cuda.shared.array(blocksize, dtype=numba.float32)
    tid = cuda.threadIdx.x  #relative idx within the block
    shared[tid] = temp
    cuda.syncthreads()

    #pair-wise parallel reduction
    stride = cuda.blockDim.x // 2
    while stride > 0:
        if tid < stride:
            shared[tid] += shared[tid + stride]
        stride //= 2
        cuda.syncthreads()

    if tid == 0:        #the partial sum within each block will be stored in idx=0
        atomicAdd(result, shared[0])        #atomic add operation to sum the partial sums


    
if __name__=="__main__":


  N = 1000
  x = np.ones(N, dtype=np.float32)
  y = np.ones(N, dtype=np.float32)
  result = np.zeros(1, dtype=np.float32)
  
  d_x = cuda.to_device(x)
  d_y = cuda.to_device(y)
  d_result = cuda.to_device(result)


  blocksize = 256
  gridsize = (N + blocksize - 1) // blocksize

  scalar_product[gridsize, blocksize](d_x, d_y, d_result, blocksize)  
  d_result.copy_to_host(result)

  if result == N:
      print("Test Passed!")
  else: 
      print("Test Failed!")




    
    


    