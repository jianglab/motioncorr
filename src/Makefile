
CPPFLAGS = -static -I./SP++3/include
CXXFLAGS = -O3
CUFLAGS = -O3 -lcuda -lcufft -arch=sm_20  -Xptxas -dlcm=ca  -lpthread -I./SP++3/include
CULIBFLAGS = cufunc.o DFAlign.o func.o mrc.o dim.o safefft.o -L./lib -lfftw3f -lm
NVCC =  nvcc


DoseFProject: func.cpp mrc.cpp
	g++ ${CPPFLAGS} ${CXXFLAGS} -c func.cpp -o func.o
	g++ ${CPPFLAGS} ${CXXFLAGS} -c mrc.cpp -o mrc.o
	g++ ${CPPFLAGS} ${CXXFLAGS} -c dim.cpp -o dim.o
	${NVCC} ${CUFLAGS} -c cufunc.cu -o cufunc.o
	${NVCC} ${CUFLAGS} -c DFAlign.cpp -o DFAlign.o
	g++ ${CXXFLAGS} -c safefft.cpp -o safefft.o
	${NVCC} ${CUFLAGS} ${CULIBFLAGS} dosefgpu_driftcorr.cpp -o ../bin/dosefgpu_driftcorr
	nvcc -O3 gpuinfo.cu -o ../bin/gpuinfo
clean:
	rm -rf *.o
