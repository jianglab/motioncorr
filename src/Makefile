
CPPFLAGS = -static -I./SP++3/include
CXXFLAGS = -O3
CUFLAGS = -O3 -lcuda -lcufft -arch=sm_20  -Xptxas -dlcm=ca  -lpthread -I./SP++3/include
CULIBFLAGS = cufunc.o DFAlign.o func.o mrc.o dim.o safefft.o -L./lib -lfftw3f -lm
NVCC =  nvcc


DoseFProject: func.o mrc.o dim.o safefft.o
	${NVCC} ${CUFLAGS} -c cufunc.cu -o cufunc.o
	${NVCC} ${CUFLAGS} -c DFAlign.cpp -o DFAlign.o
	${NVCC} ${CUFLAGS} ${CULIBFLAGS} dosefgpu_driftcorr.cpp -o ../bin/dosefgpu_driftcorr
	${NVCC} -O3 gpuinfo.cu -o ../bin/gpuinfo
clean:
	rm -rf *.o
