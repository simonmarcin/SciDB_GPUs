/**
 * @file PhysicalConvolution.cpp
 *
 * @author simon.marcin
 *
 * @brief Messy convolution algorithm to test all possible designs and parameters to
 * evaluate the best setup for GPU accelerated Operators.
 * Writes a lot of debug messages.
 *
 *
 */

#include <limits>
#include <string>
#include <sstream>
#include <vector>

#include <system/Exceptions.h>
#include <query/TypeSystem.h>
#include <query/Operator.h>
#include <util/Platform.h>
#include <util/Network.h>
#include <boost/scope_exit.hpp>

#include <log4cxx/logger.h>
#include "convolution.h"
using std::shared_ptr;
using std::make_shared;

using namespace std;

namespace scidb
{

static log4cxx::LoggerPtr logger(log4cxx::Logger::getLogger("scidb.testop"));

using namespace scidb;

template <class T>
inline void resetIterators(vector<std::shared_ptr<T> > & iterators)
{
    for (size_t i=0; i<iterators.size(); i++)
    {
        if(iterators[i].get())
        {
            iterators[i]->flush();
            iterators[i].reset();
        }
    }
}


void split(const std::string &s, char delim, std::vector<std::string> &elems) {
    std::stringstream ss;
    ss.str(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        elems.push_back(item);
    }
}

std::vector<std::string> split(const std::string &s, char delim) {
    std::vector<std::string> elems;
    split(s, delim, elems);
    return elems;
}

struct chunkEdges {
	Coordinates firstOverlap;
	Coordinates first;
	Coordinates lastOverlap;
	Coordinates last;
} ;



class ChunkQueue {
   public:
		std::vector< chunkEdges > edges;
		std::vector< size_t > nbrObjects;
		std::vector< size_t > nbrObjectsOverlaps;
		std::vector< size_t > inputSize;
		std::vector< size_t > outputSize;
		std::shared_ptr<ConstArrayIterator> srcArrayIterator;
		std::shared_ptr<ConstChunkIterator> srcChunkIterator;
		std::shared_ptr<ArrayIterator> outArrayIterator;
		std::shared_ptr<ChunkIterator> outChunkIterator;
		std::vector<int64_t> chunkInterval;
		std::vector<int64_t> chunkOverlap;
		std::vector< DimensionDesc > dimensions;
		TypeId attrType;
		TypeEnum attrEnum;
		size_t nbrChunks = 0;
		size_t nbrDimensions;
		int64_t chunkIndex = -1;

		ChunkQueue(std::shared_ptr<Array>& inputArray,
				std::shared_ptr<Array>& outputArray, size_t attributeId);

		//int64_t getNextChunk(int &overlapLenght, std::vector<std::vector<int32_t>> &chunkValues);
		template <class type> int64_t getNextChunk(int &overlapLenght, type* chunkValues);
		int writeChunk(int64_t chunkNbr, float* resultValues, std::shared_ptr<Query>& query);
		bool end();

	private:
		std::vector< chunkEdges > allCorners;
};

int ChunkQueue::writeChunk(int64_t chunkNbr, float* resultValues, std::shared_ptr<Query>& query){

	// get chunk position using chunkNbr
	chunkEdges edge = allCorners[chunkNbr];

	// create a result chunk and a value
	std::shared_ptr<ChunkIterator> outChunkIterator = outArrayIterator->newChunk(edge.first).getIterator(query,ChunkIterator::SEQUENTIAL_WRITE);
	Value value;

	// write result to resultChunck
	size_t nbrValue = 0;
	while (!outChunkIterator->end()){

		value.setFloat(resultValues[nbrValue]);
		outChunkIterator->writeItem(value);
		++(*outChunkIterator);
		nbrValue++;
	}

	outChunkIterator->flush();
	outChunkIterator.reset();

	return 0;

}

template <class type> int64_t ChunkQueue::getNextChunk(int &overlapLenght, type* chunkValues){

	//todo: check if overlapLenght is possible


	// if there is no chunk left, return -1
	LOG4CXX_INFO(logger, "getnext -> end?" );
	if(end()){return -1;}
	LOG4CXX_INFO(logger, "no end" );

	// get ChunkIterator for src_chunk
	try{
		ConstChunk const& inputChunk = srcArrayIterator->getChunk();


		bool zz = inputChunk.pin();

		srcChunkIterator = inputChunk.getConstIterator(0); // 0 = incl overlaps
		chunkIndex++;

		// get coordinates of chunks
		chunkEdges corners;
		corners.firstOverlap=inputChunk.getFirstPosition(true);
		corners.first=inputChunk.getFirstPosition(false);
		corners.lastOverlap=inputChunk.getLastPosition(true);
		corners.last=inputChunk.getLastPosition(false);
		allCorners.push_back(corners);

		// get lenghts and offsets
		int64_t xLenght = corners.lastOverlap[1] - corners.firstOverlap[1]+1;
		int64_t yLenght = corners.lastOverlap[0] - corners.firstOverlap[0]+1;
		int64_t xLenghtNoOverlap = corners.last[1] - corners.first[1]+1;
		int64_t x = corners.firstOverlap[1];
		int64_t y = corners.firstOverlap[0];
		int xOffset = (corners.firstOverlap[1]==corners.first[1]) ? overlapLenght : 0;
		int yOffset = (corners.firstOverlap[0]==corners.first[0]) ? overlapLenght : 0;
		int xShift = (corners.first[1]-corners.firstOverlap[1]>overlapLenght) ?
				corners.firstOverlap[1]+corners.first[1]-corners.firstOverlap[1]-overlapLenght :
				corners.firstOverlap[1];
		int yShift = (corners.first[0]-corners.firstOverlap[0]>overlapLenght) ?
				corners.firstOverlap[0]+corners.first[0]-corners.firstOverlap[0]-overlapLenght :
				corners.firstOverlap[0];


		// process input chunk
		while (!srcChunkIterator->end()){

			// check if coordinates are in requested range
			if(x >= corners.first[1]-overlapLenght and x<= corners.last[1]+overlapLenght){
				if(y >= corners.first[0]-overlapLenght and y<= corners.last[0]+overlapLenght){
					//LOG4CXX_INFO(logger, "write to: " << ((yOffset+y)-yShift)*(xLenghtNoOverlap+2*overlapLenght)  +  (xOffset+x)-xShift);
					chunkValues[((yOffset+y)-yShift)*(xLenghtNoOverlap+2*overlapLenght)  +  (xOffset+x)-xShift] = srcChunkIterator->getItem().get<type>();
				}
			}

			// increase x & y value
			if (x+1 >= xLenght+corners.firstOverlap[1] ){
				x=corners.firstOverlap[1];
				y++;
			}else{
				x++;
			}

			++(*srcChunkIterator);
		}

		// get next chunk
		++(*srcArrayIterator);

		inputChunk.unPin();
		return chunkIndex;


	}catch(...){
		//there is a problem in fetching the next chunk
		LOG4CXX_INFO(logger, "catch getChunk()!!!! " );
		return -1;
	}


}

bool ChunkQueue::end(){
	return srcArrayIterator->end();
}

ChunkQueue::ChunkQueue( std::shared_ptr<Array>& inputArray,
		std::shared_ptr<Array>& outputArray, size_t attributeId) {


	// get ArrayIterator
	srcArrayIterator = inputArray->getConstIterator(attributeId);
	outArrayIterator = outputArray->getIterator(attributeId);

	// get details about the attribute and the chunks
	try{
		dimensions = inputArray->getArrayDesc().getDimensions();
		nbrDimensions = dimensions.size();

		for (size_t i=0;i<nbrDimensions;i++){
			chunkInterval.push_back(dimensions[i].getChunkInterval());
			chunkOverlap.push_back(dimensions[i].getChunkOverlap());
		}

		// try to get a chunk, if it fails there are no chunks
		ConstChunk const& infoChunk = srcArrayIterator->getChunk();
		attrType = infoChunk.getAttributeDesc().getType();
		attrEnum = typeId2TypeEnum(attrType);

	}catch(...){
		LOG4CXX_INFO(logger, "no chunks" );
		nbrChunks = -1;
		return;
	}

	// reset iterator
	srcArrayIterator->reset();


}





class Physicalconvolution: public PhysicalOperator
{
public:


	// toDo: split between public and private
	bool useGPU = false;
	bool useCPU = true; // standard calculation mode
	bool debug = true;
	bool useStreams = false;
	bool async = false;
	int repetitions = 1;

    Physicalconvolution(std::string const& logicalName, std::string const& physicalName,
            Parameters const& parameters, ArrayDesc const& schema) :
                PhysicalOperator(logicalName, physicalName, parameters, schema)
    {}


    string getMainParameter() const
    {
        if (_parameters.empty())
        {
            return "arrays";
        }

        OperatorParamPhysicalExpression& exp =
            *(std::shared_ptr<OperatorParamPhysicalExpression>&)_parameters[0];
        return exp.getExpression()->evaluate().getString();
    }

    bool isCoordinateBetween(int64_t x , int64_t y, chunkEdges& edges){
    	if(x<edges.first[1] or x>edges.last[1]){return false;}
    	if(y<edges.first[0] or y>edges.last[0]){return false;}
    	return true;
    }


    size_t getCellNbr(int64_t x , int64_t y, chunkEdges& edges){
    	size_t res = (y-edges.firstOverlap[0])*(edges.lastOverlap[1]-edges.firstOverlap[1]+1);
    	res += x-edges.firstOverlap[1];
    	return res;
    }

    template <class type> void convolutionCPU(ChunkQueue& queue, std::vector<float> kernel, std::shared_ptr<Query> query)
    {

    	// init helper variables
    	LOG4CXX_INFO(logger,"start convolutionCPU");
    	int64_t chunkNbr;
		int status;
    	int kernelLength = (int) sqrt(kernel.size())/2;
    	size_t const size_y = queue.chunkInterval[0]+(kernelLength*2);
    	size_t const size_x = queue.chunkInterval[1]+(kernelLength*2);
		size_t const res_size_y = queue.chunkInterval[0];
    	size_t const res_size_x = queue.chunkInterval[1];
		size_t valueBytes = sizeof(type)*size_y*size_x;
		size_t resultBytes = sizeof(float)*res_size_y*res_size_x;
		size_t kernelBytes = sizeof(float)*kernel.size();
		size_t kernelSize = kernel.size();
		LOG4CXX_INFO(logger,"helper vars done");

    	// init memory on host and device
		type* chunkValues = new type[size_y*size_x]();
		float* resultValues = new float[res_size_y*res_size_x]();

    	// process all chunks
    	while(!queue.end()){

			// reset memory to default value (in case of edge chunks)
			std::fill(chunkValues, chunkValues + (size_x*size_y), 0);
			std::fill(resultValues, resultValues + (res_size_x*res_size_y), 0);

        	// fill host memory with next chunk from ChunkQueue
			LOG4CXX_INFO(logger,"getNextChunk");
        	chunkNbr=queue.getNextChunk(kernelLength, chunkValues);
        	LOG4CXX_INFO(logger,"getNextChunk done");

        	//loop through all coordinates
        	LOG4CXX_INFO(logger,"runConvolution");
        	for(size_t y = 0;y<res_size_y;y++){
        		for(size_t x = 0;x<res_size_x;x++){
        			float sum = 0;

        			for (int reps=0;reps<repetitions;reps++){

						for (int j = -kernelLength; j <= kernelLength; j++){
							for (int i = -kernelLength; i <= kernelLength; i++){
								sum += kernel[ ((int)kernelSize/2)  +(j*( 2*kernelLength+1 ))+i] *chunkValues[(y+j+kernelLength)*(size_x) +  x+i+kernelLength];
							}
						}

        			}

        			resultValues[y*res_size_x+x] = sum;
        		}
        	}
        	LOG4CXX_INFO(logger,"runConvolution done");

        	// write result to output chunk
        	LOG4CXX_INFO(logger,"writeChunk");
        	queue.writeChunk(chunkNbr, resultValues, query);
        	LOG4CXX_INFO(logger,"writeChunk done");


    	}
    }


    template <class type> void convolutionGPU(ChunkQueue& queue, std::vector<float> kernel, std::shared_ptr<Query> query)
    {

    	// init helper variables
    	int64_t chunkNbr[2];
		int status;
    	int kernelLength = (int) sqrt(kernel.size())/2;
    	size_t const size_y = queue.chunkInterval[0]+(kernelLength*2);
    	size_t const size_x = queue.chunkInterval[1]+(kernelLength*2);
		size_t const res_size_y = queue.chunkInterval[0];
    	size_t const res_size_x = queue.chunkInterval[1];
		size_t valueBytes = sizeof(type)*size_y*size_x;
		size_t resultBytes = sizeof(float)*res_size_y*res_size_x;
		size_t kernelBytes = sizeof(float)*kernel.size();


		// use template data type
		type* chunkValues[2];
    	chunkValues[0]= new type[size_y*size_x]();
    	if(async){chunkValues[1]= new type[size_y*size_x]();}


		float* resultValues[2];
		resultValues[0]= new float[res_size_y*res_size_x]();
		if(async){resultValues[1]= new float[res_size_y*res_size_x]();}


		// create a GPUHandler Object and pass memory
		GPUHandle <type> gpuHandler = GPUHandle<type>(kernel.data(), kernelBytes,
			chunkValues, valueBytes, resultValues, resultBytes, logger,
			res_size_x, kernel.size(), debug, useStreams, async);

		if(!async){
			// process all chunks (sync version)
			while(!queue.end()){

				// reset memory to default value (in case of edge chunks)
				std::fill(chunkValues[0], chunkValues[0] + (size_x*size_y), 0);
				std::fill(resultValues[0], resultValues[0] + (res_size_x*res_size_y), 0);

				// fill host memory with next chunk from ChunkQueue
				LOG4CXX_INFO(logger,"getNextChunk");
				chunkNbr[0]=queue.getNextChunk(kernelLength, chunkValues[0]);
				LOG4CXX_INFO(logger,"getNextChunk done");

				// copy values to memory on device
				LOG4CXX_INFO(logger,"copyChunkValues");
				gpuHandler.copyChunkValues(0);
				LOG4CXX_INFO(logger,"copyChunkValues done");

				// run kernel
				LOG4CXX_INFO(logger,"runConvolution");
				gpuHandler.runConvolution(0, repetitions);
				LOG4CXX_INFO(logger,"runConvolution done");

				// write result to output chunk
				LOG4CXX_INFO(logger,"copyResultValues");
				gpuHandler.copyResultValues(0);
				LOG4CXX_INFO(logger,"copyResultValues done");
				LOG4CXX_INFO(logger,"writeChunk");
				queue.writeChunk(chunkNbr[0], resultValues[0], query);
				LOG4CXX_INFO(logger,"writeChunk done");

			}
		}else{
			// process all chunks (async version)
			bool first = true;
			int i = 0;
			while(!queue.end()){
				if(!first){
					LOG4CXX_INFO(logger,"copyResultValuesAsync - i:"<<i);
					gpuHandler.copyResultValues(i);
					LOG4CXX_INFO(logger,"copyResultValuesAsync 'done' - i:"<<i);
				}

				// increase i mod2
				i=(i+1)%2;

				// load chunk to memory and execute copy (async) to device
				std::fill(chunkValues[i], chunkValues[i] + (size_x*size_y), 0);
				LOG4CXX_INFO(logger,"getNextChunk - i:"<<i);
				chunkNbr[i]=queue.getNextChunk(kernelLength, chunkValues[i]);
				LOG4CXX_INFO(logger,"getNextChunk done - i:"<<i);
				LOG4CXX_INFO(logger,"copyChunkValuesAsync - i:"<<i);
				gpuHandler.copyChunkValues(i);
				LOG4CXX_INFO(logger,"copyChunkValuesAsync 'done' - i:"<<i);
				LOG4CXX_INFO(logger,"runConvolutionAsync - i:"<<i);
				gpuHandler.runConvolution(i, repetitions);
				LOG4CXX_INFO(logger,"runConvolutionAsync 'done' - i:"<<i);

				if(!first){
					// wait till we have the result of the last chunk in memory
					LOG4CXX_INFO(logger,"waitEvent - i:"<<(i+1)%2);
					gpuHandler.waitEvent((i+1)%2);
					LOG4CXX_INFO(logger,"waitEvent done - i:"<<(i+1)%2);
					LOG4CXX_INFO(logger,"writeChunk - i:"<<(i+1)%2);
					queue.writeChunk(chunkNbr[(i+1)%2], resultValues[(i+1)%2], query);
					LOG4CXX_INFO(logger,"writeChunk done - i:"<<(i+1)%2);
					std::fill(resultValues[(i+1)%2], resultValues[(i+1)%2] + (res_size_x*res_size_y), 0);
				}else{
					first=false;
				}


			}
			LOG4CXX_INFO(logger,"copyResultValuesAsync - i:"<<i);
			gpuHandler.copyResultValues(i);
			LOG4CXX_INFO(logger,"copyResultValuesAsync 'done' - i:"<<i);
			LOG4CXX_INFO(logger,"waitEvent - i:"<<i);
			gpuHandler.waitEvent(i);
			LOG4CXX_INFO(logger,"waitEvent done - i:"<<i);
			LOG4CXX_INFO(logger,"writeChunk - i:"<<i);
			queue.writeChunk(chunkNbr[i], resultValues[i], query);
			LOG4CXX_INFO(logger,"writeChunk done - i:"<<i);
		}

    }



	void getKernel(std::vector<float>& kernel, std::shared_ptr<Array>& kernelArray, std::shared_ptr<Query> query)
    {
		// init values and vectors
		InstanceID const myId    = query->getInstanceID();
        InstanceID const coordId = query->getCoordinatorID() == INVALID_INSTANCE ? myId : query->getCoordinatorID();
		size_t const numInstances = query->getInstancesCount();
		//std::shared_ptr< CoordinateSet > coordSet = kernelArray->getChunkPositions();
		//size_t numChunks=(*coordSet).size();
		size_t numChunks=1;

		// if you have the chunk representing the kernel, copy it into the vector
		try {
			std::shared_ptr<ConstArrayIterator> arrayIter = kernelArray->getConstIterator(0);
			std::shared_ptr<ConstChunkIterator> chunkIter;
			int i = 0;
			while (!arrayIter->end()){
				ConstChunk const& inputChunk = arrayIter->getChunk();
				chunkIter = inputChunk.getConstIterator();
				while (!chunkIter->end()){
					kernel[i]=chunkIter->getItem().getFloat();
					i++;
					++(*chunkIter);
				}
				++(*arrayIter);
			}

		}catch(...){
			LOG4CXX_INFO(logger,"No kernel on this instance.");
		}

		//step 1: send the kernel or the init values it to the coordinator
		if(myId != coordId){ // kernel or -1 to the coordinator

			std::shared_ptr <SharedBuffer> buffer (new MemoryBuffer(kernel.data(), sizeof(float)*kernel.size()));
            BufSend(coordId, buffer, query);

			// step 4: receive kernel and assign it to vector
            buffer = BufReceive(coordId, query);
            float* ptr = static_cast<float*> (buffer->getData());
            for(size_t k =0; k<kernel.size(); ++k){
            	kernel[k] =*ptr;
            	++ptr;
            }

        }else{ // get information from nodes
			std::vector<std::vector<float>> receivedKernels(numInstances, vector<float>(kernel.size()));
			for(size_t i =0; i<numInstances; ++i){
				LOG4CXX_INFO(logger,"i:"<<i);
                if(i==myId){
                    receivedKernels[myId] = kernel;
                    continue;
                }

                shared_ptr<SharedBuffer> buffer = BufReceive(i, query);
                float* ptr = static_cast<float*> (buffer->getData());
                for(size_t k =0; k<kernel.size(); ++k){
                	receivedKernels[i][k] =*ptr;
                	++ptr;
                }
            }

			// step 2: find the kernel
			for(size_t i =0; i<numInstances; ++i){
				LOG4CXX_INFO(logger,"i:"<<i);
				for(size_t k =0; k<kernel.size(); ++k){
					if (receivedKernels[i][k] != -1){
						LOG4CXX_INFO(logger,"kernel is: "<<i);
						kernel = receivedKernels[i];
						k=kernel.size()+1;
						i=numInstances+1;
					}
				}
			}
			// step 3: send the kernel to all instances
			for(size_t i =0; i<numInstances; ++i){
				if(i==myId){continue;}
				shared_ptr<SharedBuffer> buffer(new MemoryBuffer(kernel.data(), sizeof(float)*kernel.size()));
                BufSend(i, buffer, query);
			}
		}
	}


	void setKeywords(){

		// split input string
    	string  const parameters = getMainParameter();
    	std::vector<std::string> tmp = split(parameters, ',');
    	for(int i=0;i<tmp.size();i++){
    		std::vector<std::string> tmp2 = split(tmp[i], '=');

    		// set keywords
    		if (tmp2[0] == "runOn"){
    			if(tmp2[1] == "GPU"){useGPU = true;useCPU = false;}
    			if(tmp2[1] == "CPU"){useCPU = true;}
    		}
    		if (tmp2[0] == "GPUStreams"){
    			if(tmp2[1] == "true"){useStreams = true;}
    		}
    		if (tmp2[0] == "async"){
    			if(tmp2[1] == "true"){async = true;useStreams = true;}
    		}
    		if (tmp2[0] == "repetitions"){
    			repetitions = std::stoi(tmp2[1]);
    		}

    	}

	}

    shared_ptr<Array> execute(vector<std::shared_ptr<Array> >& inputArrays, std::shared_ptr<Query> query)
    {
    	//todo: extract keywords from query
    	bool debug = true;

    	// extract keywords and set flags
    	setKeywords();

		// init of Array and Chunk Iterators for each attribute
    	LOG4CXX_INFO(logger,"attrs");
		size_t nAttrs = inputArrays[0]->getArrayDesc().getAttributes().size();
		nAttrs-=1; //There is always 1 attribute too much

		// init of Array and Chunk Iterators for Output Array (same number of attributes)
		std::shared_ptr<Array> outputArray(new MemArray(_schema, query));

		// init a ChunkQueue Object
		ChunkQueue queue(inputArrays[0],outputArray,0);
		LOG4CXX_INFO(logger,"queue init ok");

		// get kernel from instance which is holding it (as kernel = 1 chunk)
		size_t kernelSize = inputArrays[1]->getArrayDesc().getSize();
		std::vector<float> kernel(kernelSize,-1);
		getKernel(kernel, inputArrays[1], query);
		LOG4CXX_INFO(logger,"get kernel ok");

		// if this instance has no input chunks then return
		if (queue.nbrChunks == -1) {
			LOG4CXX_INFO(logger,"I have nothing to do. Bye.");
			return outputArray;
		}

		switch(queue.attrEnum)
		{
		case TypeEnum::TE_INT32:
			LOG4CXX_INFO(logger,"Template: TE_INT32.");
			if(useCPU){ convolutionCPU<int32_t>(queue, kernel, query); }
			if(useGPU){	convolutionGPU<int32_t>(queue, kernel, query); }
			break;
		case TypeEnum::TE_INT16:
			LOG4CXX_INFO(logger,"Template: TE_INT16.");
			if(useCPU){ convolutionCPU<int16_t>(queue, kernel, query); }
			if(useGPU){	convolutionGPU<int16_t>(queue, kernel, query); }
			break;
		default:
			//throw USER_EXCEPTION(SCIDB_SE_EXECUTION, "Not supported data type.");
			break;
		}


		//return Output Array
		return outputArray;

    }

};

REGISTER_PHYSICAL_OPERATOR_FACTORY(Physicalconvolution, "convolution", "Physicalconvolution");

}
 // end namespace scidb
