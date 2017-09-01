/**
 * @file PhysicalFirdem.cpp
 *
 * @author simon.marcin
 *
 * @brief "main" of plugin. Init all needed helpers and executes processing loop.
 *
 * @tparam t_streams number of used GPU streams
 *
 */

#include <limits>
#include <string>
#include <sstream>
#include <vector>
#include <stdlib.h>
#include <chrono>
#include <thread>

#include <system/Exceptions.h>
#include <query/TypeSystem.h>
#include <query/Operator.h>
#include <util/Platform.h>
#include <util/Network.h>
#include <boost/scope_exit.hpp>

#include <log4cxx/logger.h>
#include "firdem.h"
#include "../../utils/DataAccess.h"
#include <cuda_runtime.h>

using std::shared_ptr;
using std::make_shared;


// Standard parameters - fixed after running tests and running benchmarks
#define __precision double
#define __QueueLenght 3


using namespace std;
namespace scidb
{

static log4cxx::LoggerPtr logger(log4cxx::Logger::getLogger("scidb.firdem"));


// split function for strings
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


class PhysicalFirdem: public PhysicalOperator
{

private:

	// Use more than one GPU (default false)
	bool multiGPU = false;
	// number of GPUs running on the nodes (default 1)
	int nbrGPUs = 1;
	// calculate median temperature (default true)
	bool calc_EMWMT = true;
	// upper limit of temperature range (bins)
	double logtmax = 7.5;
	// lower limit of temperature range (bins)
	double logtmin = 5.5;
	
public:

    PhysicalFirdem(std::string const& logicalName, std::string const& physicalName,
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


	void setKeywords(){

		// split input string
    	string  const parameters = getMainParameter();
    	std::vector<std::string> tmp = split(parameters, ',');
    	for(int i=0;i<tmp.size();i++){
    		std::vector<std::string> tmp2 = split(tmp[i], '=');

    		// set keywords
    		if (tmp2[0] == "multiGPU"){
    			int tmp = std::stoi(tmp2[1]);
    			if (tmp > 1 && tmp <= 16)
    				nbrGPUs  = tmp;
    				multiGPU = true;
    		}
    		if (tmp2[0] == "EMWMT"){
    			int tmp = std::stoi(tmp2[1]);
    			if (tmp == 0)
    				calc_EMWMT = false;
    		}
    		 if (tmp2[0] == "logtmax"){
    			double tmp = std::stod(tmp2[1]);
    			if (tmp > 3.0 && tmp < 10.0)
    				logtmax = tmp;
    		}
    		if (tmp2[0] == "logtmin"){
    			double tmp = std::stod(tmp2[1]);
    			if (tmp < logtmax)
    				logtmin = tmp;
    		}
    	}
	}


	template <class t_precision, class t_input, size_t t_queueLenght>
	std::shared_ptr<Array> doWork(vector<std::shared_ptr<Array> >& inputArrays, std::shared_ptr<Query> query){

		LOG4CXX_INFO(logger, "firdem start" );

		// we process all AIA wavelenghts stored in SciDB
		size_t nAttrs = inputArrays[0]->getArrayDesc().getAttributes().size();
		nAttrs-=1; //There is always 1 attribute too much

		// init Output Array
		std::shared_ptr<Array> outputArray(new MemArray(_schema, query));

		DataAccess<t_input,t_precision,t_queueLenght> data(inputArrays[0],outputArray,query,0,1,nAttrs);
		LOG4CXX_INFO(logger,"queue init ok");
		t_precision** tmp_out_arr = data.initOutputMemory();
		t_input* inputData[t_queueLenght];

		Dem<t_precision, t_input, 32, 6, t_queueLenght> gpuDEM(data.getChunkSize(),calc_EMWMT);
		LOG4CXX_INFO(logger,"gpuDEM created");

		//init all streams
		bool workToDo = true;
		bool tmp[t_queueLenght];
		bool finished[t_queueLenght];

		for(size_t i = 0;i<t_queueLenght;i++){
			finished[i]=false;
			inputData[i] = data.getChunk(i);
			if(inputData[i] == nullptr){
				finished[i]=true;
				continue;
			}
			gpuDEM.runKernel(inputData[i],tmp_out_arr[i],i);
		}

		// process all chunks - on any free stream
		while(workToDo){
			//Check each stream
			for(size_t i = 0;i<t_queueLenght;i++){
				// is the stream already finished?
				if(!finished[i]){
					// is the kernel done?
					if(gpuDEM.isStreamFinished(i)){
						// store results
						data.setChunk(i,tmp_out_arr[i]);
						// get next chunck
						inputData[i] = data.getChunk(i);
						// check if we still have work to do
						if(inputData[i] == nullptr){
							finished[i]=true;
							tmp[i]=true;
							for(size_t j = 0;j<t_queueLenght;j++){
								tmp[i] &= finished[j];
							}
							workToDo = !tmp[i];
						}else{
							gpuDEM.runKernel(inputData[i],tmp_out_arr[i],i);
						}
					}
				}
			}
			std::this_thread::sleep_for(std::chrono::milliseconds(5));
		}

		//return Output Array
		LOG4CXX_INFO(logger, "firdem done" );
		return outputArray;

	}


    shared_ptr<Array> execute(vector<std::shared_ptr<Array> >& inputArrays, std::shared_ptr<Query> query)
    {

    	// extract keywords and set flags and variables
    	setKeywords();

    	// set cuda devices
    	InstanceID const myId    = query->getInstanceID();
    	if (multiGPU){
    		cudaSetDevice(myId % nbrGPUs);
    	}else{
    		cudaSetDevice(0);
    	}
    	//cudaCheckErrors("cudaSetDevice failed");

    	// get the first chunk of the array to choose the datatype
    	TypeEnum attrEnum;
    	try{
    		std::shared_ptr<ConstArrayIterator> srcArrayIterator =
    				inputArrays[0]->getConstIterator(0);
    		ConstChunk const& infoChunk = srcArrayIterator->getChunk();
    		TypeId attrType = infoChunk.getAttributeDesc().getType();
    		attrEnum = typeId2TypeEnum(attrType);
    	}catch(...){
    		LOG4CXX_INFO(logger, "No Chunks on this instance, return an empty Array." );
    		std::shared_ptr<Array> outputArray(new MemArray(_schema, query));
    		return outputArray;
    	}

		// evaluate templating
		switch(attrEnum)
		{
		case TypeEnum::TE_INT32:
			LOG4CXX_INFO(logger,"Template: TE_INT32.");
			return doWork<__precision, int32_t, __QueueLenght>(inputArrays, query);
			break;
		case TypeEnum::TE_INT16:
			LOG4CXX_INFO(logger,"Template: TE_INT16.");
			return doWork<__precision, int16_t, __QueueLenght>(inputArrays, query);
			break;
		default:
			//throw USER_EXCEPTION(SCIDB_SE_EXECUTION, "Not supported data type.");
			break;
		}


    }

};

REGISTER_PHYSICAL_OPERATOR_FACTORY(PhysicalFirdem, "firdem", "PhysicalFirdem");

}

// explicit instantiation of tempalte
template class Dem<double,int16_t,32,6,__QueueLenght>;
