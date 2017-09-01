### DataAccess
Abstraction layer between SciDB and GPU code. Prepares reused pinned memory block for data transfer. Loads multiple (default 3) chunks concurrent to pipeline more work to the GPU.


### GPUHandler
Base class for GPU algorithms. Takes care GPU streams (default 3) to occupy all three engines on the GPU at the same time.


### intTabulated
Implementation of IDL int_tabulated (and therefore also spl_interp and spl_init). Used to get the same results as in IDL to verify code.


### cudaInverse
Inverse function for small arrays in CUDA as template header-only function.