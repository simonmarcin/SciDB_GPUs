### SciDB Plugins
Plugins used to analyse the usage of GPUs with SciDB plugins.

#### utils
Use these helper and base classes to write GPU accelerated SciDB plugins. Abstracts the SciDB part from the GPU kernel and manages concurrent asynchronous copys and kernel execution as well as reads/writes data to SciDB. See DEM for an implemenation.

#### GPU accelerated DEM algorithm
Adapted version of the DEM algorithm of Plowman, et al. Runs each pixel in a thread block to solve the iverse problem. Uses a lot of different strategies to hide overhead of the GPU and of SciDB. See section benchmarks for performance analysis.

#### GPU accelerated Convolution algorithm
Algorithm used to test different approaches to hide overhead and different designs. It is only for benchmarking and debugging purposes.


