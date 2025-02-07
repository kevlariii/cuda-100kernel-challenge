from numba import cuda
import numpy as np

@cuda.jit
def add_vectors(x, y, result):
  assert len(x) == len(y), "The two vectors should have the same size"
  thread_position = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

  if thread_position < len(x):
    result[thread_position] = x[thread_position] + y[thread_position]
 
def test_kernel(result, expected):
  assert np.allclose(result, expected), "Test Failed!"
  print("Test Passed!")

if __name__=="__main__":


  N = 1000
  x = np.ones(N)
  y = np.zeros(N)
  result = np.zeros(N)
  
  d_x = cuda.to_device(x)
  d_y = cuda.to_device(y)
  d_result = cuda.to_device(result)


  threads_per_block = 256
  gridsize = (N + threads_per_block - 1) // threads_per_block

  add_vectors[gridsize, threads_per_block](d_x, d_y, d_result)
  
  d_result.copy_to_host(result)

  expected = np.ones(N)
  test_kernel(result, expected)