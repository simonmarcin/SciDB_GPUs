/**
 * @file PhysicalConvolution.cpp
 *
 * @author simon.marcin
 *
 * @brief GPU Convolution class to examine different setups, parameters and designs.
 *
 *
 */

#include <cuda_runtime.h>
#include <log4cxx/logger.h>


#ifndef DATE_H
#define DATE_H



template <class type> class GPUHandle
{
private:
	float* kernel;
	size_t kernelBytes;
	size_t smBytes;
	type* chunkValues[2];
	size_t valueBytes;
	float* resultValues[2];
	size_t resultBytes;
	log4cxx::LoggerPtr logger;
	size_t chunckWidth;
	size_t kernelSize;
	int overlap;
	type* d_Input[2];
    float* d_Output[2];
    float* d_Kernel;
    bool debug;
    bool useStreams;
    bool async;

    cudaStream_t streams[2];
    cudaEvent_t syncEvents[2];

public:
    ~GPUHandle();
    GPUHandle(float* in_kernel, size_t in_kernelBytes,type* in_chunkValues[],
			size_t in_valueBytes, float* in_resultValues[], size_t in_resultBytes,
			log4cxx::LoggerPtr logger, size_t in_chunckWidth, size_t in_kernelSize, bool in_debug,
			bool in_useStreams, bool in_async);

    void runConvolution(int, int);
    void copyChunkValues(int);
    void copyResultValues(int);
    void waitEvent(int);

};



#endif
