#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "dim.h"

#include "mrc.h"


#include <complex>
#include <vector>
#include <string>
using namespace std;

struct CPLX
{
	float x;
	float y;
};

CPLX cXc(CPLX a, CPLX b);

void fft2d(float* buf, DIM nsam);
void ifft2d(float* buf, DIM nsam);

void SetFastFFT(float *buf, DIM nsam);
void ReleaseFastFFT();
void fft2d_fast();
void ifft2d_fast();


void buf2fft(float *buf, float *fft, DIM nsam);
void fft2buf(float *buf, float *fft, DIM nsam);
void buf2mrc(const char *filename, float* buf, int nx, int ny, int nz);

DIM verifyCropSize(int nx, int ny, int offsetx, int offsety, DIM nsamout, int FutureBin);
DIM crop(float *bufin, DIM nsamin, float *bufout, int offsetx, int offsety, DIM nsamout, int FutureBin=1);  //return new nsam, may different from nsamout
DIM crop2fft(float *bufin, DIM nsamin, float *bufout, int offsetx, int offsety, DIM nsamout, int FutureBin=1);

void MinMaxMean(float *buf, size_t size, float &min, float &max, float &mean);
float STD(float *buf, size_t size, float mean);

void shift2d_phase(float *buf, DIM nsam, float sx, float sy);
void CPLXAdd(CPLX *dst, CPLX *src, int sizec);

void FFTDataToLogDisp(float *pIn, float *pOut, int dim);

void rmNoiseCC(float* hboxmap, int box, int nframe, int peakR);
void cosmask2d(complex<float> *pfft, int nsam);

void histMinMax(float *buf, int size, float &min, float &max, float threshold=0.005);
void buf2Disp(char *disp, float *buf, int size);
void buf2DispShort(short *disp, float *buf, int size);

void FFTModulusToDispBuf(float *pIn, float *pOut, DIM nsam);
void BinFFTDispBufToChar(char *pDisp, int dispdim, float *pfft, DIM nsam);
void BinFFTDispBufToChar(short *pDisp, int dispdim, float *pfft, DIM nsam);
void DispFFTToDispShort(short *pDisp, float *pfft, int dispdim);

float* ReadRef(const char *filename, int dst_nx, int dst_ny);
bool ApplyRef(float* im, bool bDark, float* dark, bool bGain, float* gain, size_t size);

void FlipYAxis(float *buf, int nx, int ny);
