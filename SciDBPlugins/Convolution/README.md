# GPU accelerated Convolution algorithm
Runs a convolution kernel on an array. This implementation is not for use in production! It supports all kind of different approaches to run the GPU kernel. It is used to examine all possible parameters and to find the most efficient design. 


# Example
```
$ iquery -naq "convolution(aia_array,gauss_blur_11x11,'runOn=GPU,async=true,GPUStreams=true')"
Query was executed successfully

See UserInterface or Benchmark for more details.
```


# Arguments
runOn=[GPU/CPU]: Either run on CPU or GPU<br>
GPUStreams=[true/false]: Use multiple GPUStreams to pipeline work.<br>
asynv=[true/false]: run asyc copys from Host to Device and vice versa.<br>
repetitions=N: Run N repetitions - this is only used to simulate high work load.


# Installation
Compile the plugin with make.
Copy the libconvolution.so to all of your SciDB nodes Plugin folder.
