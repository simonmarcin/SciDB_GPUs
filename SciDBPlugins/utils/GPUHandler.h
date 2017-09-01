/**
 * @file GPUHandler.h
 *
 * @author simon.marcin
 *
 * @brief Base class for GPUAlgorithms used with SciDB and
 * multiple streams to process multiple chunks async.
 *
 * @tparam t_streams number of used GPU streams
 *
 */

#include <cuda_runtime.h>
#include <stdio.h>

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

typedef CUstream_st * cudaStream_t;
typedef CUevent_st * cudaEvent_t;


template <size_t t_streams>
class GPUHandler{

public:

	// check if stream[i] is finished (if syncEvent is reached)
	bool isStreamFinished(size_t stream);

	// constructor
	GPUHandler();

	// destructor
	virtual ~GPUHandler();

protected:

	// get CUDA stream
	cudaStream_t getStream(size_t stream);

	// register a SyncEvent on stream[i]
	void setSyncEvent(size_t stream);

private:

	// array of streams
	cudaStream_t streams[t_streams];

	// array of SyncEvents
	cudaEvent_t syncEvents[t_streams];

};


template <size_t t_streams>
void GPUHandler<t_streams>::setSyncEvent(size_t stream){
	cudaEventRecord(syncEvents[stream],streams[stream]);
	cudaCheckErrors("cudaEventRecord");
}


template <size_t t_streams>
bool GPUHandler<t_streams>::isStreamFinished(size_t stream){
	// check if kernel of stream is already finished
	if(cudaEventQuery(syncEvents[stream]) == 0){
		return true;
	}
	return false;
}


template <size_t t_streams>
cudaStream_t GPUHandler<t_streams>::getStream(size_t stream){
	return streams[stream];
}


template <size_t t_streams>
GPUHandler<t_streams>::GPUHandler(){
	// init streams and syncEvents
	for (size_t i=0; i < t_streams; i++){
		streams[i] = cudaStream_t();
		cudaStreamCreateWithFlags(&(streams[i]),cudaStreamNonBlocking);
		cudaCheckErrors("Create Stream.");
		syncEvents[i] = cudaEvent_t();
		cudaEventCreateWithFlags(&(syncEvents[i]), cudaEventDisableTiming);
		cudaCheckErrors("Create Event.");
	}
	cudaCheckErrors("Init of streams and syncEvents.");
}


template <size_t t_streams>
GPUHandler<t_streams>::~GPUHandler(){
	// delete streams and syncEvents
	for (size_t i=0; i < t_streams; i++){
		cudaStreamDestroy(streams[i]);
		cudaEventDestroy(syncEvents[i]);
	}
	cudaCheckErrors("Delete streams and syncEvents");
}
