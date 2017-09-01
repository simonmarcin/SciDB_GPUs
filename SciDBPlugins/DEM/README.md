# GPU accelerated DEM algorithm
Returns 32 DEM coefficients, the chi2 error and if requested the emission weighted median temperature (EMWMT) and total emission measure (EMTOT).


# Example
```
$ iquery -naq "firdem(aia_array,'EMWMT=1,MultiGPU=8')"
Query was executed successfully

See UserInterface or Benchmark for more details.
```

# Arguments
EMWMT=[1/0]: Calculate EMWMT and EMTOT
MultiGPU=[1-N]: Number of GPU cards to use
logtmax: upper limit of temperature bins
logtmin: lower limit of temperature bins


# Installation
Compile the plugin with make.
Copy the libfirdem.so to all of your SciDB nodes Plugin folder.
