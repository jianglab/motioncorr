#include "safefft.h"
#include "fftw3.h"
#include <pthread.h>

pthread_mutex_t mutex_fftwplan;

void initFFTWLock()
{
	pthread_mutex_init(&mutex_fftwplan,NULL);
}
void freeFFTWLock()
{
	pthread_mutex_destroy(&mutex_fftwplan);
}

void fft2d_safe(float* buf, DIM nsam)
{
	pthread_mutex_lock(&mutex_fftwplan);
	fftwf_plan plan_fft=fftwf_plan_dft_r2c_2d(nsam.y,nsam.x,buf,reinterpret_cast<fftwf_complex *>(buf),FFTW_ESTIMATE);
	pthread_mutex_unlock(&mutex_fftwplan);

	fftwf_execute(plan_fft);

	pthread_mutex_lock(&mutex_fftwplan);
	fftwf_destroy_plan(plan_fft);
	pthread_mutex_unlock(&mutex_fftwplan);
}


void ifft2d_safe(float* buf, DIM nsam)
{
	pthread_mutex_lock(&mutex_fftwplan);
	fftwf_plan plan_fft=fftwf_plan_dft_c2r_2d(nsam.y,nsam.x,reinterpret_cast<fftwf_complex *>(buf),buf,FFTW_ESTIMATE);
	pthread_mutex_unlock(&mutex_fftwplan);

	fftwf_execute(plan_fft);

	pthread_mutex_lock(&mutex_fftwplan);
	fftwf_destroy_plan(plan_fft);
	pthread_mutex_unlock(&mutex_fftwplan);
}
