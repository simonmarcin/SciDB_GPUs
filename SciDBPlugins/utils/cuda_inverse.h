/**
 * @file cudaInverse.h
 *
 * @author simon.marcin
 *
 * @brief (L)^-1 code of ZhengzhongSun adapted as device function and only for
 * small matrices. Thread block has to be at least t_n x t_n.
 *
 * @tparam t_precision data type (float or double) of input and output matrix.
 * @tparam t_n size of a square matrix n x n.
 *
 * @param A input matrix of size t_n x t_n
 * @param I output matrix of size t_n x t_n
 *
 */

template <typename t_precision, size_t t_n>
__device__ __forceinline__ void Inverse(t_precision A[], t_precision I[]){

	// (L)^-1 code of ZhengzhongSun adapted as device function and only for small matrices
	//https://github.com/ZhengzhongSun/Matrix-Inversion-with-CUDA/blob/master/matrixInversion_gpu.cu

	//Init diag matrix
	if (threadIdx.x < t_n && threadIdx.y < t_n){
		I[threadIdx.y*t_n + threadIdx.x] = 0.0;
		if (threadIdx.x == threadIdx.y){
			I[threadIdx.y*t_n + threadIdx.x] = 1.0;
		}
	}
	__syncthreads();

	#pragma unroll
	for (int i = 0; i<t_n; i++){
		//nodiag_normalize
		if (threadIdx.x < t_n && threadIdx.y < t_n){
			if (threadIdx.x == i && threadIdx.x!=threadIdx.y){
				I[threadIdx.x*t_n + threadIdx.y] /= A[i*t_n + i];
				A[threadIdx.x*t_n + threadIdx.y] /= A[i*t_n + i];
			}
		}
		__syncthreads();


		//diag_normalize
		if (threadIdx.x < t_n && threadIdx.y < t_n){
			if (threadIdx.x == threadIdx.y && threadIdx.x == i){
				I[threadIdx.x*t_n + threadIdx.y] /= A[i*t_n + i];
				A[threadIdx.x*t_n + threadIdx.y] /= A[i*t_n + i];
			}
		}
		__syncthreads();


		//gaussjordan
		if (threadIdx.x < t_n && threadIdx.y < t_n){
			if (threadIdx.x != i){
				I[threadIdx.x*t_n + threadIdx.y] -= I[i*t_n + threadIdx.y] * A[threadIdx.x*t_n + i];
				if (threadIdx.y != i){
					A[threadIdx.x*t_n + threadIdx.y] -= A[i*t_n + threadIdx.y] * A[threadIdx.x*t_n + i];
				}
			}
		}
		__syncthreads();

		//set_zero
		if (threadIdx.x < t_n && threadIdx.y < t_n){
			if (threadIdx.x != i){
				if (threadIdx.y == i){
					A[threadIdx.x*t_n + threadIdx.y] = 0;
				}
			}
		}
		__syncthreads();

	}
}
