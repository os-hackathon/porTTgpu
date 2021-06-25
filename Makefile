
CXX=hipcc
CXXFLAGS=-O0 -g -L/usr/local/cuda -lcublas -lcublasLt -L/usr/lib64/libcutensor/11 -lcutensor
HIP_PLATFORM=nvidia

ifeq ($(HIP_PLATFORM), nvidia)
  ARCH=--gpu-architecture=sm_60
endif


kernels.o: kernels.cpp
	${CXX} -c ${CXXFLAGS} ${ARCH} kernels.cpp -o $@

main.o: main.cpp
	${CXX} -c ${CXXFLAGS} main.cpp -o $@

demo: main.o kernels.o
	${CXX} ${CXXFLAGS} *.o -o $@
