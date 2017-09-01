/**
 * @file convolution.cu
 *
 * @author simon.marcin
 *
 * @brief Messy convolution algorithm to test all possible designs and parameters to
 * evaluate the best setup for GPU accelerated Operators.
 * Can run synchronous copys or asynchronous, can run multiple streams or not.
 * Writes a lot of debug messages.
 *
 *
 */

#include <stdio.h>
#include "convolution.h"
#include <iostream>
#include <float.h>

#define DSIZE 1024
#define DVAL 10
#define nTPB 256
#define cudaHostRegisterPortable 0x01
#define ROWS_BLOCKDIM_X 16
#define ROWS_BLOCKDIM_Y 16


#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)



template <class type> __global__ void convolutionKernel(float* d_Output,type* d_Input, size_t chunckWidth,
		int overlap, size_t kernelSize, float* d_Kernel, int repetitions)
{
	// define shared memory
	const int globalWidth = chunckWidth+(2*overlap);
    //__shared__ float s_Data[34][34];
    extern __shared__ float s_Data[];

    // offset to overlaps (halo elements)
    const int globalX = (blockIdx.x * ROWS_BLOCKDIM_X) + threadIdx.x;
    const int globalY = (blockIdx.y * ROWS_BLOCKDIM_Y) + threadIdx.y;

    // load inner chunk elements
    s_Data[(threadIdx.y+overlap)*(ROWS_BLOCKDIM_X+2*overlap) + threadIdx.x+overlap] = d_Input[((globalY+overlap)*globalWidth)+globalX+overlap];
    //printf("thread %d,%d, input=%d to=%d,%d\n", threadIdx.y,threadIdx.x, d_Input[((globalY+overlap)*globalWidth)+globalX+overlap], threadIdx.y+overlap,threadIdx.x+overlap);

    // // load upper halo elements
    if (threadIdx.y < overlap){
    	s_Data[(threadIdx.y)*(ROWS_BLOCKDIM_X+2*overlap) + threadIdx.x+overlap]=d_Input[globalY*globalWidth + globalX+overlap];
    }
    //load bottom halo elements
    if (threadIdx.y >= ROWS_BLOCKDIM_Y-overlap){
    	s_Data[(threadIdx.y+2*overlap)*(ROWS_BLOCKDIM_X+2*overlap) + threadIdx.x+overlap]=d_Input[(globalY+2*overlap)*globalWidth + globalX+overlap];
    }
    // load left halo elements
    if (threadIdx.x < overlap){
    	s_Data[(threadIdx.y+overlap)*(ROWS_BLOCKDIM_X+2*overlap) + threadIdx.x]=d_Input[(globalY+overlap)*globalWidth + globalX];
    }
    // load right halo elements
    if (threadIdx.x >= ROWS_BLOCKDIM_X-overlap){
    	s_Data[(threadIdx.y+overlap)*(ROWS_BLOCKDIM_X+2*overlap) + threadIdx.x+2*overlap]=d_Input[(globalY+overlap)*globalWidth + globalX+2*overlap];
    }

    // left upper edges
    if (threadIdx.x < overlap and threadIdx.y < overlap){
    	s_Data[(threadIdx.y)*(ROWS_BLOCKDIM_X+2*overlap) + threadIdx.x]=d_Input[(globalY)*globalWidth + globalX];
    }
    // right upper edges
    if (threadIdx.x >= ROWS_BLOCKDIM_X-overlap and threadIdx.y < overlap){
    	s_Data[(threadIdx.y)*(ROWS_BLOCKDIM_X+2*overlap) + threadIdx.x+2*overlap]=d_Input[(globalY)*globalWidth + globalX+2*overlap];
    }
    // left bottom edges
    if (threadIdx.x < overlap and threadIdx.y >= ROWS_BLOCKDIM_Y-overlap){
    	s_Data[(threadIdx.y+2*overlap)*(ROWS_BLOCKDIM_X+2*overlap) + threadIdx.x]=d_Input[(globalY+2*overlap)*globalWidth + globalX];
    }
    // right bottom edges
    if (threadIdx.x >= ROWS_BLOCKDIM_X-overlap and threadIdx.y >= ROWS_BLOCKDIM_Y-overlap){
    	s_Data[(threadIdx.y+2*overlap)*(ROWS_BLOCKDIM_X+2*overlap) + threadIdx.x+overlap]=d_Input[(globalY+2*overlap)*globalWidth + globalX+2*overlap];
    }

    //Compute and store results
    __syncthreads();


    float sum = 0;

    for (int reps=0;reps<repetitions;reps++) {

		#pragma unroll
		for (int j = -overlap; j <= overlap; j++){
			#pragma unroll
			for (int i = -overlap; i <= overlap; i++){
				//sum += s_Data[(threadIdx.y+j+overlap)*(ROWS_BLOCKDIM_X+2*overlap) + threadIdx.x+i+overlap];
				sum += d_Kernel[ ((int)kernelSize/2)  +(j*( 2*overlap+1 ))+i] * s_Data[(threadIdx.y+j+overlap)*(ROWS_BLOCKDIM_X+2*overlap) +  threadIdx.x+i+overlap];
			}
		}
    }

	d_Output[globalY*chunckWidth+globalX] = sum;

}

// init the template function as the CUDA compiler doesn't know anything about SciDB
template __global__ void convolutionKernel<int16_t>(float*,int16_t*, size_t,int, size_t, float*, int);
template __global__ void convolutionKernel<int32_t>(float*,int32_t*, size_t,int, size_t, float*, int);


template <class type> void GPUHandle<type>::copyChunkValues(int i){

	// transfer data to device
	if(debug){LOG4CXX_INFO(logger, "start memory transfer!");}

	if(useStreams){
		if(async){
			cudaMemcpyAsync(d_Input[i], chunkValues[i], valueBytes, cudaMemcpyHostToDevice,(streams[i]));
			cudaCheckErrors("cudaMemcpyAsync (async): chunkValues");
		}else{
			cudaMemcpyAsync(d_Input[0], chunkValues[0], valueBytes, cudaMemcpyHostToDevice,(streams[0]));
			cudaCheckErrors("cudaMemcpyAsync (streams): chunkValues");
		}
	}else{
		cudaMemcpy(d_Input[0], chunkValues[0], valueBytes, cudaMemcpyHostToDevice);
		cudaDeviceSynchronize();
		cudaCheckErrors("cudaMemcpy: chunkValues");
	}

    if(debug){LOG4CXX_INFO(logger, "memory transfer done!");}

}

template <class type> void GPUHandle<type>::waitEvent(int i){
	cudaEventSynchronize(syncEvents[i]);
	cudaCheckErrors("cudaEventSynchronize");
}

template <class type> void GPUHandle<type>::copyResultValues(int i){

	// transfer data to host
	if(debug){LOG4CXX_INFO(logger, "start memory transfer!");}
	if(useStreams){
		if(async){
			cudaMemcpyAsync(resultValues[i], d_Output[i], resultBytes, cudaMemcpyDeviceToHost, (streams[i]));
			// Inject a sync Event as we need to be sure that we have the result
			cudaEventRecord(syncEvents[i],streams[i]);
			cudaCheckErrors("cudaMemcpyAsync (async): d_Output");
		}else{
			cudaMemcpyAsync(resultValues[0], d_Output[0], resultBytes, cudaMemcpyDeviceToHost, (streams[0]));
			cudaStreamSynchronize((streams[0]));
			cudaCheckErrors("cudaMemcpyAsync (streams): d_Output");
		}
	}else{
		cudaMemcpy(resultValues[0], d_Output[0], resultBytes, cudaMemcpyDeviceToHost);
		cudaCheckErrors("cudaMemcpy: d_Output");
	}
	if(debug){LOG4CXX_INFO(logger, "memory transfer done!");}

}

template <class type> void GPUHandle<type>::runConvolution(int i, int repetitions){

	// define the kernel grid (2D)

    dim3 blocks(max(chunckWidth / ROWS_BLOCKDIM_X,size_t(1)), max(chunckWidth / ROWS_BLOCKDIM_Y,size_t(1)));
    dim3 threads(ROWS_BLOCKDIM_X, ROWS_BLOCKDIM_Y);

   // run kernel
    if(debug){LOG4CXX_INFO(logger, chunckWidth<< "  "<<overlap<< "  "<<kernelSize);}
    if(useStreams){
    	if(async){
    		convolutionKernel<<<blocks, threads, smBytes, (streams[i])>>>(d_Output[i],d_Input[i],chunckWidth,overlap,kernelSize, d_Kernel, repetitions);
			cudaCheckErrors("convolutionRowsKernel() (streams)");
    	}else{
    		convolutionKernel<<<blocks, threads, smBytes, (streams[0])>>>(d_Output[0],d_Input[0],chunckWidth,overlap,kernelSize, d_Kernel, repetitions);
			cudaCheckErrors("convolutionRowsKernel() (streams)");
    	}
    }else{
    	if(debug){LOG4CXX_INFO(logger, "blocks: " <<  chunckWidth / ROWS_BLOCKDIM_X << "," << chunckWidth / ROWS_BLOCKDIM_Y << "," << smBytes);}
		convolutionKernel<<<blocks, threads, smBytes>>>(d_Output[0],d_Input[0],chunckWidth,overlap,kernelSize, d_Kernel, repetitions);
		cudaDeviceSynchronize();
		cudaCheckErrors("convolutionRowsKernel()");
    }
    if(debug){LOG4CXX_INFO(logger, "kernel done!");}



}

template <class type> GPUHandle<type>::~GPUHandle(){

	// unpin memory
	cudaHostUnregister(chunkValues[0]);
	cudaCheckErrors("cudaHostUnRegister: chunkValues");
	cudaHostUnregister(resultValues[0]);
	cudaCheckErrors("cudaHostUnRegister: resultValues");
	if(async){
		cudaHostUnregister(chunkValues[1]);
		cudaCheckErrors("cudaHostUnRegister: chunkValues[1]");
		cudaHostUnregister(resultValues[1]);
		cudaCheckErrors("cudaHostUnRegister: resultValues[1]");
	}

	// free memory on device
	cudaFree(d_Input[0]);
	cudaCheckErrors("cudaFree: d_Input");
	cudaFree(d_Output[0]);
	cudaCheckErrors("cudaFree: d_Output");
	cudaFree(d_Kernel);
	cudaCheckErrors("cudaFree: d_Kernel");
	if(async){
		cudaFree(d_Input[1]);
		cudaCheckErrors("cudaFree async: d_Input");
		cudaFree(d_Output[1]);
		cudaCheckErrors("cudaFree async: d_Output");
	}

	// destroy streams
	if(useStreams){
		cudaStreamDestroy(streams[0]);
		cudaStreamDestroy(streams[1]);
	}

	// destroy sync events
	if(async){
		cudaEventDestroy(syncEvents[0]);
		cudaCheckErrors("cudaEventDestroy: syncEvents[0]");
		cudaEventDestroy(syncEvents[1]);
		cudaCheckErrors("cudaEventDestroy: syncEvents[1]");
	}
}


template <class type> GPUHandle<type>::GPUHandle(float* in_kernel, size_t in_kernelBytes,type* in_chunkValues[],
		size_t in_valueBytes, float* in_resultValues[], size_t in_resultBytes,
		log4cxx::LoggerPtr in_logger, size_t in_chunckWidth, size_t in_kernelSize, bool in_debug,
		bool in_useStreams, bool in_async){

	// save values
	kernel = in_kernel;
	kernelBytes = in_kernelBytes;
	chunkValues[0] = in_chunkValues[0];
	chunkValues[1] = in_chunkValues[1];
	valueBytes = in_valueBytes;
	resultValues[0] = in_resultValues[0];
	resultValues[1] = in_resultValues[1];
	resultBytes = in_resultBytes;
	logger = in_logger;
	overlap = (int) sqrt(in_kernelSize)/2;
	kernelSize = in_kernelSize;
	chunckWidth = in_chunckWidth;
	smBytes = (ROWS_BLOCKDIM_X+2*overlap)*(ROWS_BLOCKDIM_Y+2*overlap)*sizeof(float);
	debug = in_debug;
	useStreams = in_useStreams;
	async = in_async;

	// create streams
	if(useStreams){
		streams[0] = cudaStream_t();
		cudaStreamCreateWithFlags(&(streams[0]),cudaStreamNonBlocking);
		cudaCheckErrors("cudaStreamCreateWithFlags: streams[0]");
		streams[1] = cudaStream_t();
		cudaStreamCreateWithFlags(&(streams[1]),cudaStreamNonBlocking);
		cudaCheckErrors("cudaStreamCreateWithFlags: streams[1]");
	}

	// create sync events
	if(async){
		syncEvents[0] = cudaEvent_t();
		syncEvents[1] = cudaEvent_t();
		cudaEventCreateWithFlags(&(syncEvents[0]), cudaEventDisableTiming);
		cudaCheckErrors("cudaEventCreateWithFlags: syncEvents[0]");
		cudaEventCreateWithFlags(&(syncEvents[1]), cudaEventDisableTiming);
		cudaCheckErrors("cudaEventCreateWithFlags: syncEvents[1]");
	}

	// pin host memory (this is slow but we only do it once to allow async transfers)
	cudaHostRegister(chunkValues[0], valueBytes, cudaHostRegisterPortable);
	cudaCheckErrors("cudaHostRegister: chunkValues[0]");
	cudaHostRegister(resultValues[0], resultBytes, cudaHostRegisterPortable);
	cudaCheckErrors("cudaHostRegister: resultValues[0]");
	if(async){
		cudaHostRegister(chunkValues[1], valueBytes, cudaHostRegisterPortable);
		cudaCheckErrors("cudaHostRegister: chunkValues[1]");
		cudaHostRegister(resultValues[1], resultBytes, cudaHostRegisterPortable);
		cudaCheckErrors("cudaHostRegister: resultValues[1]");
	}

	// prepare Memory on device
	cudaMalloc((void **)&(d_Input[0]), valueBytes);
	cudaCheckErrors("cudaMalloc: d_Input");
	cudaMalloc((void **)&(d_Output[0]), resultBytes);
	cudaCheckErrors("cudaMalloc: d_Output");
	cudaMalloc((void **)&d_Kernel, kernelBytes);
	cudaCheckErrors("cudaMalloc: d_Kernel");
	if(async){
		cudaMalloc((void **)&(d_Input[1]), valueBytes);
		cudaCheckErrors("cudaMalloc async: d_Input");
		cudaMalloc((void **)&(d_Output[1]), resultBytes);
		cudaCheckErrors("cudaMalloc async: d_Output");
	}

	// copy kernel to constant memory
	//cudaMemcpyToSymbol(c_Kernel, kernel, kernelBytes);

	// copy kernel to global memory (as the size is not known at compilation time)
	if(useStreams){
		cudaMemcpyAsync(d_Kernel, kernel, kernelBytes, cudaMemcpyHostToDevice, streams[0]);
	    cudaCheckErrors("cudaMemcpyAsync (streams): d_Kernel");
	}else{
	    cudaMemcpy(d_Kernel, kernel, kernelBytes, cudaMemcpyHostToDevice);
	    cudaCheckErrors("cudaMemcpy: d_Kernel");
	}

}

// init the template classes as nvcc doesn't know the used types
template class GPUHandle<int16_t>;
template class GPUHandle<int32_t>;


