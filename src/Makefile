
CPPFLAG = -static -O3 -I./SP++3/include
CUFLAG = -O3 -lcuda -lcufft -arch=sm_20  -Xptxas -dlcm=ca  -lpthread -I./SP++3/include
CULIBFLAG = cufunc.o DFAlign.o func.o mrc.o dim.o safefft.o -L./lib -lfftw3f -lm
NVCC =  nvcc


DoseFProject: func.cpp mrc.cpp
	g++ ${CPPFLAG} -c func.cpp -o func.o 
	g++ ${CPPFLAG} -c mrc.cpp -o mrc.o
	g++ ${CPPFLAG} -c dim.cpp -o dim.o
	${NVCC} ${CUFLAG} -c cufunc.cu -o cufunc.o
	${NVCC} ${CUFLAG} -c DFAlign.cpp -o DFAlign.o
	g++ -O3 -c safefft.cpp -o safefft.o
	${NVCC} ${CUFLAG} ${CULIBFLAG} dosefgpu_driftcorr.cpp -o ../bin/dosefgpu_driftcorr
	nvcc -O3 gpuinfo.cu -o ../bin/gpuinfo	
clean:
	rm -rf *.o
