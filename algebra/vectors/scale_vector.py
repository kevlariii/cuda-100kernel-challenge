from numba import cuda
import numpy as np

@cuda.jit
def scale_vector(x, alpha):
  if alpha == 0: return np.zeros(len(x))

  thread_position = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

  if thread_position < len(x):
    x[thread_position] = alpha * x[thread_position]
 
def test_kernel(result, expected):
  assert np.allclose(result, expected), "Test Failed!"
  print("Test Passed!")

if __name__=="__main__":


  N = 1000
  x = np.ones(N)
  alpha = 5
  
  
  d_x = cuda.to_device(x)

  threads_per_block = 256
  gridsize = (N + threads_per_block - 1) // threads_per_block

  scale_vector[gridsize, threads_per_block](d_x, alpha)
  
  d_x.copy_to_host(x)

  expected = np.ones(N) * alpha
  test_kernel(x, expected)