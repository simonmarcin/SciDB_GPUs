/**
 * @file DataAccess.h
 *
 * @author simon.marcin
 *
 * @brief Uses SciDB array and chunk iterators and writes data to
 * pinned system memory for GPU usage. Size of queue should be the
 * same as number of used streams in GPUHandler.
 *
 * @tparam t_input data type of input
 * @tparam t_output data type of result
 * @tparam t_queue_size number of parallel open chunks
 *
 */

#include <system/Exceptions.h>
#include <query/TypeSystem.h>
#include <query/Operator.h>
#include <util/Platform.h>
#include <util/Network.h>

#include <log4cxx/logger.h>
#include <cuda_runtime.h>

static log4cxx::LoggerPtr logger(log4cxx::Logger::getLogger("scidb.dataAccess"));


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


template <class t_input, class t_output, size_t t_queue_size>
class DataAccess
{
   public:

		// Default constructor
		DataAccess(std::shared_ptr<scidb::Array>& inputArray,
				std::shared_ptr<scidb::Array>& outputArray,
				std::shared_ptr<scidb::Query>& query, size_t overlap,
				size_t groupByAttribute, size_t nAttr);

		// create pinned memory for output values
		t_output** initOutputMemory();

		// Destructor
		~DataAccess();

		// get next Data Chunk for queueSpot (read from SciDB)
		t_input* getChunk(size_t queueSpot);

		// set Data Chunk for queueSpot (write to SciDB)
		int16_t setChunk(size_t queueSpot, t_output* resultValues);

		// get chunk_size
		int64_t getChunkSize();

	private:

		// number of elements of each chunk
		int64_t chunk_size;
		// vector of input array iterators
		std::vector<std::shared_ptr<scidb::ConstArrayIterator> > srcArrayIterators;
		// vector of input chunk iterators
		std::vector<std::shared_ptr<scidb::ConstChunkIterator> > srcChunkIterators;
		// vector of output array iterators
		std::vector<std::shared_ptr<scidb::ArrayIterator> > outArrayIterators;
		// vector of output chunk iterators
		std::vector<std::shared_ptr<scidb::ChunkIterator> > outChunkIterators;
		// number of input Attributes (0 to x)
		size_t numOfAttributes;
		// number of output Attributes (0 to x)
		size_t numOfAttributesOutput;
		// vector of positions for each QueueSpot
		scidb::Coordinates chunkCoordinates[t_queue_size];
		// query of SciDB
		std::shared_ptr<scidb::Query> query;
		// do we have to take care of overlaps?
		size_t overlap;
		// defines the input and output structure of the flat memory area
		size_t groupByAttribute;
		// pinned memory area for input data
		t_input* inputDataPinned[t_queue_size];
		// pinned memory for output values
		t_output* outputDataPinned[t_queue_size];

		// get next Data Chunk for queueSpot without overlaps (read from SciDB)
		t_input* getChunkNoOverlap(size_t queueSpot);

		// get next Data Chunk for queueSpot with overlaps (read from SciDB)
		t_input* getChunkOverlap(size_t queueSpot);


};


template <class t_input, class t_output, size_t t_queue_size>
int16_t DataAccess<t_input,t_output,t_queue_size>::setChunk(size_t queueSpot, t_output* resultValues){


	// get ChunkIterator for each Attribute
	for(size_t i=0;i<numOfAttributesOutput;i++){
		outChunkIterators[i] = outArrayIterators[i]->newChunk(
				chunkCoordinates[queueSpot]).getIterator(query,
				scidb::ChunkIterator::SEQUENTIAL_WRITE);
	}

	// tmp counter for result memory area
	int64_t tmp_counter = 0;
	scidb::Value value;

	// loop through the data and copy the data to the new SciDb chunk
	if(groupByAttribute == 1){

		// a1,a2,a3 for all x1; a1,a2,a3 for all x2
		while (!outChunkIterators[0]->end()){
			for (size_t i=0;i<numOfAttributesOutput;i++){
				value.set<t_output> (resultValues[tmp_counter]);
				outChunkIterators[i]->writeItem(value);
				++(*outChunkIterators[i]);
				++tmp_counter;
			}
		}

	}else{

		// x1,x2,x3 for all a1; x1,x2,x3 for all a2
		for (size_t i=0;i<numOfAttributes;i++){
			while (!srcChunkIterators[i]->end()){
				value.set<t_output>(resultValues[tmp_counter]);
				outChunkIterators[i]->writeItem(value);
				++(*outChunkIterators[i]);
				++tmp_counter;
			}
		}

	}

	for(size_t i=0;i<numOfAttributesOutput;i++){
		outChunkIterators[i]->flush();
		outChunkIterators[i].reset();
	}

	// return 0 when everything went well
	return 0;

}

template <class t_input, class t_output, size_t t_queue_size>
int64_t DataAccess<t_input,t_output,t_queue_size>::getChunkSize() {
	return chunk_size;
}

template <class t_input, class t_output, size_t t_queue_size>
t_input* DataAccess<t_input,t_output,t_queue_size>::getChunkNoOverlap(size_t queueSpot)
{

	// get ChunkIterator for each Attribute
	for(size_t i=0;i<numOfAttributes;i++){
		scidb::ConstChunk const& inputChunk = srcArrayIterators[i]->getChunk();
		srcChunkIterators[i] = inputChunk.getConstIterator(scidb::ConstChunkIterator::IGNORE_OVERLAPS);//ChunkIterator::IGNORE_OVERLAPS
		++(*srcArrayIterators[i]);
	}

	// save position of chunk
	chunkCoordinates[queueSpot]  = srcChunkIterators[0]->getPosition();

	// counter for inputDataPinned memory area
	int64_t tmp_counter = 0;

	// loop through the data and copy the data to the pinned memory area for GPU usage
	if(groupByAttribute == 1){

		// a1,a2,a3 for all x1; a1,a2,a3 for all x2
		while (!srcChunkIterators[0]->end()){
			for (size_t i=0;i<numOfAttributes;i++){
				(inputDataPinned[queueSpot])[tmp_counter] = srcChunkIterators[i]->getItem().get<t_input>();
				++(*srcChunkIterators[i]);
				++tmp_counter;
			}
		}

	}else{

		// x1,x2,x3 for all a1; x1,x2,x3 for all a2
		for (size_t i=0;i<numOfAttributes;i++){
			tmp_counter = 0;
			while (!srcChunkIterators[i]->end()){
				(inputDataPinned[queueSpot])[tmp_counter] = srcChunkIterators[i]->getItem().get<t_input>();
				++(*srcChunkIterators[i]);
				++tmp_counter;
			}
		}

	}

	// return pointer to pinned memory area
	return inputDataPinned[queueSpot];

}



template <class t_input, class t_output, size_t t_queue_size>
t_input* DataAccess<t_input,t_output,t_queue_size>::getChunkOverlap(size_t queueSpot)
{

	//todo: implement the possibility to only use a part of the overlaps


	// get ChunkIterator for each Attribute
	for(size_t i=0;i<numOfAttributes;i++){
		scidb::ConstChunk const& inputChunk = srcArrayIterators[i]->getChunk();
		srcChunkIterators[i] = inputChunk.getConstIterator();//ChunkIterator::IGNORE_OVERLAPS
		++(*srcArrayIterators[i]);
	}

	// save position of chunk
	chunkCoordinates[queueSpot]  = srcChunkIterators[0]->getPosition();

	// counter for inputDataPinned memory area
	int64_t tmp_counter = 0;

	// loop through the data and copy the data to the pinned memory area for GPU usage
	if(groupByAttribute == 1){

		// a1,a2,a3 for all x1; a1,a2,a3 for all x2
		while (!srcChunkIterators[0]->end()){
			for (size_t i=0;i<numOfAttributes;i++){
				(inputDataPinned[queueSpot])[tmp_counter] = srcChunkIterators[i]->getItem().get<t_input>();
				++(*srcChunkIterators[i]);
				++tmp_counter;
			}
		}

	}else{

		// x1,x2,x3 for all a1; x1,x2,x3 for all a2
		for (size_t i=0;i<numOfAttributes;i++){
			tmp_counter = 0;
			while (!srcChunkIterators[i]->end()){
				(inputDataPinned[queueSpot])[tmp_counter] = srcChunkIterators[i]->getItem().get<t_input>();
				++(*srcChunkIterators[i]);
				++tmp_counter;
			}
		}

	}

	// return pointer to pinned memory area
	return inputDataPinned[queueSpot];

}


template <class t_input, class t_output, size_t t_queue_size>
t_input* DataAccess<t_input,t_output,t_queue_size>::getChunk(size_t queueSpot)
{

	// if there is no data left, return NullPointer
	if(srcArrayIterators[0]->end()){
		return nullptr;
	}

	//choose correspondoing subfunction
	if(overlap == 0){
		return getChunkNoOverlap(queueSpot);
	}else{
		return getChunkOverlap(queueSpot);
	}

}


template <class t_input, class t_output, size_t t_queue_size>
t_output** DataAccess<t_input,t_output,t_queue_size>::initOutputMemory(){

	for (size_t i=0;i<t_queue_size;i++){
		cudaMallocHost(&(outputDataPinned[i]),sizeof(t_output)*chunk_size*numOfAttributesOutput);
	}
	cudaCheckErrors("cudaMallocHost of output pinned Data");
	return outputDataPinned;

}


template <typename t_input, typename t_output, size_t t_queue_size>
DataAccess<t_input,t_output,t_queue_size>::DataAccess(std::shared_ptr<scidb::Array>& inputArray,
		std::shared_ptr<scidb::Array>& outputArray, std::shared_ptr<scidb::Query>& queryIn,
		size_t overlapIn, size_t groupByAttributeIn, size_t nAttr)
{

	  LOG4CXX_INFO(logger,"init Data Access");

	  // init of Input Array Iterators
	  numOfAttributes = nAttr;
	  for (size_t i=0;i<numOfAttributes;i++){
		  srcArrayIterators.push_back(inputArray->getConstIterator(i));
	  }

	  // load the first chunk of the first attribute to extract some information
	  scidb::ConstChunk const& infoChunk = srcArrayIterators[0]->getChunk();
	  if(overlap == 0){
		  chunk_size = infoChunk.getNumberOfElements(false);
	  }else{
		  chunk_size = infoChunk.getNumberOfElements(true);
	  }
	  srcArrayIterators[0]->reset();

	  // init of Output Array Iterators
	  numOfAttributesOutput = outputArray->getArrayDesc().getAttributes().size();
	  for (size_t i=0;i<numOfAttributesOutput;i++){
		  outArrayIterators.push_back(outputArray->getIterator(i));
	  }

	  // resize chunk vectors
	  srcChunkIterators.resize(numOfAttributes);
	  outChunkIterators.resize(numOfAttributesOutput);

	  // store config from input
	  query=queryIn;
	  overlap = overlapIn;
	  groupByAttribute = groupByAttributeIn;

	  // init of pinned memory area
	  cudaCheckErrors("cudaMallocHost of input pinned Data");
	  for (size_t i=0;i<t_queue_size;i++){
		  cudaHostAlloc(&(inputDataPinned[i]),sizeof(t_input)*chunk_size*numOfAttributes,cudaHostAllocPortable);
	  }
	  cudaCheckErrors("cudaMallocHost of input pinned Data");


	  LOG4CXX_INFO(logger,"init Data Access done");

}

template <typename t_input, typename t_output, size_t t_queue_size>
DataAccess<t_input,t_output,t_queue_size>::~DataAccess()
{
	  for (size_t i=0;i<t_queue_size;i++){
		  cudaFreeHost(inputDataPinned[i]);
		  cudaFreeHost(outputDataPinned[i]);
	  }
	  LOG4CXX_INFO(logger,"cleanup Data Access");
}







