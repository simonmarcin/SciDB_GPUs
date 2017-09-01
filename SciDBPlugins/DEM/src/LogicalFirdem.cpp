/**
 * @file LogicalFirdem.cpp
 *
 * @author simon.marcin
 *
 * @brief Logical SciDB operator firdem. Handels input of query and
 * creates output array definition.
 *
 */

#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include <query/Operator.h>
#include <log4cxx/logger.h>


// hardcoded number of result points: 32 (as optimal for GPU)
#define __nb2 32


using std::shared_ptr;
using boost::algorithm::trim;
using boost::starts_with;
using boost::lexical_cast;
using boost::bad_lexical_cast;

using namespace std;

namespace scidb {

class LogicalFirdem: public LogicalOperator {
public:
	LogicalFirdem(const std::string& logicalName, const std::string& alias) :
        LogicalOperator(logicalName, alias){

        ADD_PARAM_INPUT();
        ADD_PARAM_CONSTANT("string");
    }


    // output schema of firdem
    ArrayDesc inferSchema(std::vector<ArrayDesc> schemas,
            shared_ptr<Query> query){

    	// create output array with size of nb2
    	Attributes outputAttributes;
    	std::string attributeName = "t_";
    	for(size_t i=0;i<__nb2;i++){
    		outputAttributes.push_back( AttributeDesc(0, (attributeName +  std::to_string(i)),
    				TID_DOUBLE, AttributeDesc::IS_NULLABLE, 0));
    	}
    	// add some more output information (chi2 error, emtot, emwmt)
    	outputAttributes.push_back( AttributeDesc(0, "chi2", TID_DOUBLE, AttributeDesc::IS_NULLABLE, 0));
    	outputAttributes.push_back( AttributeDesc(0, "emtot", TID_DOUBLE, AttributeDesc::IS_NULLABLE, 0));
    	outputAttributes.push_back( AttributeDesc(0, "emwmt", TID_DOUBLE, AttributeDesc::IS_NULLABLE, 0));

    	// No overlaps in the resulting array
    	std::vector<DimensionDesc> outputDimensions = schemas[0].getDimensions();
    	for(size_t i = 0; i< outputDimensions.size();i++){
    		outputDimensions[i].setChunkOverlap(0);
    	}

        ArrayDesc res(
                "dem_coffs",
                outputAttributes,
                outputDimensions,
                schemas[0].getDistribution(),
                schemas[0].getResidency());
        return res;
    }
};


REGISTER_LOGICAL_OPERATOR_FACTORY(LogicalFirdem, "firdem");

}
 // end namespace scidb
