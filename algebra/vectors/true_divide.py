from numba import cuda
import numpy as np

@cuda.jit
def true_divide(x, y, result):
  thread_position = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

  if thread_position < len(x):
    result[thread_position] = x[thread_position] / y[thread_position]



def test_kernel(result, expected):
  assert np.allclose(result, expected), "Test Failed!"
  print("Test Passed!")

if __name__=="__main__":


  N = 1000
  x = np.ones(N, dtype=np.float32)*2
  y = np.ones(N, dtype=np.float32)*2
  result = np.zeros(N, dtype=np.float32)
  
  d_x = cuda.to_device(x)
  d_y = cuda.to_device(y)
  d_result = cuda.to_device(result)


  threads_per_block = 256
  gridsize = (N + threads_per_block - 1) // threads_per_block

  true_divide[gridsize, threads_per_block](d_x, d_y, d_result)  
  d_result.copy_to_host(result)

  expected = np.ones(N, dtype=np.float32)
  test_kernel(result, expected)