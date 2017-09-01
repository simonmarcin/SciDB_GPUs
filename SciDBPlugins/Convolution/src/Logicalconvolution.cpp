/**
 * @file LogicalConvolution.cpp
 *
 * @author simon.marcin
 *
 * @brief Logical operator implementation of Convolution algorithm.
 *
 */

#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include <query/Operator.h>
#include <log4cxx/logger.h>

using std::shared_ptr;
using boost::algorithm::trim;
using boost::starts_with;
using boost::lexical_cast;
using boost::bad_lexical_cast;

using namespace std;

namespace scidb {

class Logicalconvolution: public LogicalOperator {
public:
    Logicalconvolution(const std::string& logicalName, const std::string& alias) :
        LogicalOperator(logicalName, alias)
    {
        ADD_PARAM_INPUT();
        ADD_PARAM_INPUT();
        ADD_PARAM_CONSTANT("string");
    }

    /**
     * output schema of testop is same as the input schema
     */
    ArrayDesc inferSchema(std::vector<ArrayDesc> schemas,
            shared_ptr<Query> query)
    {
    	Attributes outputAttributes;
    	outputAttributes.push_back( AttributeDesc(0, "t0", TID_FLOAT, AttributeDesc::IS_NULLABLE, 0));

    	// ToDo: Is there a way SciDB handles the overlap data?
    	std::vector<DimensionDesc> outputDimensions = schemas[0].getDimensions();
    	for(size_t i = 0; i< outputDimensions.size();i++){
    		outputDimensions[i].setChunkOverlap(0);
    	}

        ArrayDesc res(
                schemas[0].getName(),
                outputAttributes,
                outputDimensions,
                schemas[0].getDistribution(),
                schemas[0].getResidency());
        return res;
    }
};

REGISTER_LOGICAL_OPERATOR_FACTORY(Logicalconvolution, "convolution");

}
 // emd namespace scidb
