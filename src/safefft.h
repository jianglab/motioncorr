#pragma once
#include "dim.h"

void initFFTWLock();
void freeFFTWLock();
void fft2d_safe(float* buf, DIM nsam);
void ifft2d_safe(float* buf, DIM nsam);
