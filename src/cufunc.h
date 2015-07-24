#pragma once

#include <cuda.h>
#include <cufft.h>
#include <string>
#include <vector>
#include "dim.h"
using namespace std;

struct MASK
{
	int x;
	int y;
	float z;
};

bool initGPU(int GPUNum);
bool ResetGPU();
void siginthandler(int param);
int getGPUList(vector<string> &namelist);
void GPUMemCheck(size_t &theFree, size_t &theTotal);

bool GPUMemAlloc(void **buf, int size);
bool GPUMemZero(void **buf, int size);
bool GPUMemFree(void **buf);
bool GPUMemH2D(void *dst, void *src, int size);
bool GPUMemD2H(void *dst, void *src, int size);
bool GPUMemD2D(void *dst, void *src, int size);
bool GPUCrop2d(float *src, DIM nsamin, float *dst, DIM offset, DIM nsamout);
bool GPUMemBinD2H(float *dst, float *src, DIM dst_nsam, DIM src_nsam);
bool GPUMemBinD2D(float *dst, float *src, DIM dst_nsam, DIM src_nsam);
void GPUFFTErrorMessage(cufftResult r, char *name);
cufftHandle GPUFFTPlan(DIM nsam);
cufftHandle GPUIFFTPlan(DIM nsam);
bool GPUFFT2d(float* dfft, cufftHandle plan);
bool GPUIFFT2d(float* dfft, cufftHandle plan);
void GPUFFTDestroy(cufftHandle &plan);
bool GPUSync();
void GPUAdd(float *dst, float *src, int size);
void GPUMultiplyNum(float *dst, float num, int size);

void GPUFFTLogModulus(float *dMod, float *dfft, DIM nsam, float scale);
void GPUFFTModulus(float *dMod, float *dfft, DIM nsam);
DIM GPURectFFTLogModulus(float *dfft, float *dsum, float *dtmp1, float *dtmp2, DIM nsam, float scale, cufftHandle hfft);
void GPUBinFFT(float *dst, int dispdim, float *src, DIM nsamsub, cufftHandle hfft);

//void MkPosList(int3 *list, int nsam, float inner_r, float outer_r);
void MkPosList(MASK *list, DIM nsam, float bfactor);
void GPUShiftCC(float *dfft, float *dsum, MASK *dposlist, float sx, float sy, DIM nsam);
void GPUShift(float *dfft, MASK *dposlist, float sx, float sy, DIM nsam);
float FindShift(float *dsrc,DIM nsam, float* hboxmap, int box, float &sx, float &sy, int wNoise=1); //wNoise==-1 to disable it

void GPUbuf2mrc(const char *filename, float* dbuf, int nx, int ny, int nz);
void testCUFFT();
