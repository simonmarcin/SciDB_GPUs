ifeq ($(SCIDB),) 
  X := $(shell which scidb 2>/dev/null)
  ifneq ($(X),)
    X := $(shell dirname ${X})
    SCIDB := $(shell dirname ${X})
  endif
endif

# A way to set the 3rdparty prefix path that is convenient
# for SciDB developers.
ifeq ($(SCIDB_VER),)
  SCIDB_3RDPARTY = $(SCIDB)
else
  SCIDB_3RDPARTY = /opt/scidb/$(SCIDB_VER)
endif

# A better way to set the 3rdparty prefix path that does
# not assume an absolute path. You can still use the above
# method if you prefer.
ifeq ($(SCIDB_THIRDPARTY_PREFIX),)
  SCIDB_THIRDPARTY_PREFIX := $(SCIDB_3RDPARTY)
endif


CFLAGS=-W -Wextra -Wall -Wno-unused-parameter -Wno-variadic-macros -Wno-strict-aliasing \
-Wno-long-long -Wno-unused -fPIC -D_STDC_FORMAT_MACROS -Wno-system-headers -isystem -O3 -g -DNDEBUG -D_STDC_LIMIT_MACROS

INC=-I. -DPROJECT_ROOT="\"$(SCIDB)\"" -I"$(SCIDB_THIRDPARTY_PREFIX)/3rdparty/boost/include/" -I"$(SCIDB)/include" -I/usr/local/cuda-8.0/include
LIBS=--shared -Wl,-soname=libfirdem.so -L. -L"$(SCIDB_THIRDPARTY_PREFIX)/3rdparty/boost/lib" -L"$(SCIDB)/lib" -Wl,-rpath,$(SCIDB)/lib:$(RPATH) -lm -L/usr/local/cuda/lib64 -lcudart -lcudadevrt -llog4cxx

SRCS=plugin.cpp 
# Compiler settings for SciDB version >= 15.7
ifneq ("$(wildcard /usr/bin/g++-4.9)","")
  CC := "/usr/bin/gcc-4.9"a
  CXX := "/usr/bin/g++-4.9"
  CFLAGS+=-std=c++11 -DCPP11
else
  ifneq ("$(wildcard /opt/rh/devtoolset-3/root/usr/bin/gcc)","")
   CC := "/opt/rh/devtoolset-3/root/usr/bin/gcc"
   CXX := "/opt/rh/devtoolset-3/root/usr/bin/g++"
   CFLAGS+=-std=c++11 -DCPP11 
  endif
endif
SRCS+= LogicalFirdem.cpp PhysicalFirdem.cpp link.o firdem.o AIAConstants.o 

all: link.o firdem.o AIAConstants.o 
	@if test ! -d "$(SCIDB)"; then echo  "Error. Try:\n\nmake SCIDB=<PATH TO SCIDB INSTALL PATH>"; exit 1; fi
	$(CXX) $(CFLAGS) $(INC) -o libfirdem.so $(SRCS) $(LIBS)
	@echo "Now copy *.so to your SciDB lib/scidb/plugins directory and run"
	@echo "iquery -aq \"load_library('firdem')\" # to load the plugin."
	@echo
	@echo "Re-start SciDB if the plugin was already loaded previously."
	@echo "Remember to copy the plugin to all your nodes in the cluster."
	
	
link.o: firdem.cu firdem.h AIAConstants.cpp AIAConstants.h
	/usr/local/cuda-8.0/bin/nvcc -m64 -O3 -Xptxas -O3 -std=c++11 -gencode arch=compute_60,code=sm_60 -dc  -Xcompiler '-fPIC'  firdem.cu AIAConstants.cpp -I. -llog4cxx
	/usr/local/cuda-8.0/bin/nvcc -m64 -O3 -Xptxas -O3 -std=c++11 -gencode arch=compute_60,code=sm_60 -Xcompiler '-fPIC' -dlink firdem.o AIAConstants.o -o link.o -llog4cxx
	
test:
	@./test.sh
clean:
	rm -f *.so *.o
