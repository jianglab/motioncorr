#include "func.h"
#include "fftw3.h"
#include <pthread.h>
#include "safefft.h"

#pragma comment(lib,"libfftw3f-3.lib")
fftwf_plan plan_fft_fast;
fftwf_plan plan_ifft_fast;


void fft2d(float* buf, DIM nsam)
{
	fftwf_plan plan_fft=fftwf_plan_dft_r2c_2d(nsam.y,nsam.x,buf,reinterpret_cast<fftwf_complex *>(buf),FFTW_ESTIMATE);  
	fftwf_execute(plan_fft);
	fftwf_destroy_plan(plan_fft);
}


void ifft2d(float* buf, DIM nsam)
{
	fftwf_plan plan_fft=fftwf_plan_dft_c2r_2d(nsam.y,nsam.x,reinterpret_cast<fftwf_complex *>(buf),buf,FFTW_ESTIMATE); 
	fftwf_execute(plan_fft);	
	fftwf_destroy_plan(plan_fft);
}



void SetFastFFT(float *buf, DIM nsam)
{
	plan_fft_fast=fftwf_plan_dft_r2c_2d(nsam.y,nsam.x,buf,reinterpret_cast<fftwf_complex *>(buf),FFTW_ESTIMATE); 
	plan_ifft_fast=fftwf_plan_dft_c2r_2d(nsam.y,nsam.x,reinterpret_cast<fftwf_complex *>(buf),buf,FFTW_ESTIMATE); 
}
void ReleaseFastFFT()
{
	fftwf_destroy_plan(plan_fft_fast);
	fftwf_destroy_plan(plan_ifft_fast);
}
void fft2d_fast()
{
	fftwf_execute(plan_fft_fast);
}
void ifft2d_fast()
{
	fftwf_execute(plan_ifft_fast);
}

CPLX cXc(CPLX a, CPLX b)
{
	CPLX c;
	c.x=a.x*b.x-a.y*b.y;
	c.y=a.x*b.y+a.y*b.x;
	return c;
}


DIM verifyCropSize(int nx, int ny, int offsetx, int offsety, DIM nsamout, int FutureBin)
{
	if(offsetx<0 || offsetx>=nx || offsety<0 || offsety>=ny || FutureBin<=0) return DIM(0,0);

	
	if(nsamout.x<=0) nsamout.x=nx-offsetx;
	if(nsamout.y<=0) nsamout.y=ny-offsety;

	if((nx-offsetx)<nsamout.x) nsamout.x=nx-offsetx;
	if((ny-offsety)<nsamout.y) nsamout.y=ny-offsety;
	
	nsamout=nsamout/(2*FutureBin)*(2*FutureBin);

	return nsamout;
}

DIM crop(float *bufin, DIM nsamin, float *bufout, int offsetx, int offsety, DIM nsamout, int FutureBin)
{
	if(bufin==0 || bufout==0) return DIM(0,0);

	nsamout=verifyCropSize(nsamin.width(),nsamin.height(),offsetx, offsety, nsamout,FutureBin);
	
	if(nsamout.x==0 || nsamout.y==0 ) return DIM(0,0);

	int i;
	for(i=0;i<nsamout.y;i++)
	{
		memcpy(bufout+i*nsamout.x, bufin+(i+offsety)*nsamin.x+offsetx, sizeof(float)*nsamout.x);
	}

	return nsamout;

}


DIM crop2fft(float *bufin, DIM nsamin, float *bufout, int offsetx, int offsety, DIM nsamout, int FutureBin)
{
	if(bufin==0 || bufout==0) return DIM(0,0);

	nsamout=verifyCropSize(nsamin.width(),nsamin.height(), offsetx, offsety, nsamout, FutureBin);
	//memset(bufout,0,(nsamout+2)*nsamout*sizeof(float));
	if(nsamout.x==0 || nsamout.y==0) return DIM(0,0);

	int i;
	for(i=0;i<nsamout.height();i++)
	{
		memcpy(bufout+i*(nsamout.width()+2), bufin+(i+offsety)*nsamin.width()+offsetx, 
				sizeof(float)*nsamout.width());
	}

	return nsamout;

}


void buf2mrc(const char *filename, float* buf, int nx, int ny, int nz)
{
	MRC mrc;
	mrc.open(filename,"wb");
	mrc.createMRC(buf,nx,ny,nz);
	mrc.close();
}

void buf2fft(float *buf, float *fft, DIM nsam)
{
	int nsamb=nsam.x+2;
	int i;
	for(i=0;i<nsam.y;i++)
	{
		memcpy(fft+i*nsamb,buf+i*nsam.x,sizeof(float)*nsam.x);
	}
}

void fft2buf(float *buf, float *fft, DIM nsam)
{
	int nsamb=nsam.x+2;
	int i;
	for(i=0;i<nsam.y;i++)
	{
		memcpy(buf+i*nsam.x,fft+i*nsamb,sizeof(float)*nsam.x);
	}
}


void MinMaxMean(float *buf, size_t size, float &min, float &max, float &mean)
{
	if(buf==NULL || size<=0) return; 
	min=1e20;
	max=-1e20;
	double dmean=0.0;
	
	size_t i;
	for(i=0;i<size;i++)
	{
		if(min>buf[i]) min=buf[i];
		if(max<buf[i]) max=buf[i];
		dmean+=buf[i];
	}
	mean=dmean/size;
}

float STD(float *buf, size_t size, float mean)
{
	if(buf==NULL || size<=0) return 0; 
	
	double std=0.0;
	double val;
	size_t i;
	for(i=0;i<size;i++)
	{
		val=buf[i]-mean;
		std+=val*val;
	}
	
	return sqrt(std/size);
}

void shift2d_phase(float *buf, DIM nsam, float sx, float sy)
{
	CPLX *bufc=(CPLX *)buf;
	
	CPLX pshift;
	float shift;
	int hnsamx=nsam.x/2;
	int hnsamy=nsam.y/2;
	int hnsamxb=hnsamx+1;

	int i,j,jj,id,is;
	float shx=sx*6.2831852/nsam.x;
	float shy=sy*6.2831852/nsam.y;
	for(j=0;j<hnsamy;j++)
	{
		id=j*hnsamxb;
		is=(j+hnsamy)*hnsamxb;
		jj=j-hnsamy;
		for(i=0;i<hnsamxb;i++)
		{
			shift=i*shx+j*shy;
			pshift.x=cos(shift);
			pshift.y=sin(shift);
			bufc[id+i]=cXc(bufc[id+i],pshift);
			
			shift=i*shx+jj*shy;
			pshift.x=cos(shift);
			pshift.y=sin(shift);
			bufc[is+i]=cXc(bufc[is+i],pshift);
		}
	}
}


void CPLXAdd(CPLX *dst, CPLX *src, int sizec)
{
	for(int i=0;i<sizec;i++)
	{
		dst[i].x+=src[i].x;
		dst[i].y+=src[i].y;
	}
}


void FFTDataToLogDisp(float *pIn, float *pOut, int dim)
{
	//convert fft_complex to Intensity
	int fftsize=(dim+2)*dim/2;
	int hdim=dim/2;
	int width=(dim+2)/2;
	int id0=dim*dim/2+dim/2;
	int id1=dim*dim/2+dim/2;
	int id2=dim/2-dim*dim/2;
	int id3=3*dim*dim/2+dim/2;
	//pIn[0]=0;
	float absval;
	int i,j,is;

	int index;
	float val;
	float fs=0.00005;

	for(i=0;i<dim;i++)
	{
		is=i*width;
		for(j=0;j<hdim;j++)
		{
			absval=pIn[is+j];
			val=log(1+fs*absval);

			index=i*dim+j;
			if(i<hdim) pOut[id0+index]=pOut[id1-index]=val;
			else if(i>hdim) pOut[id2+index]=pOut[id3-index]=val;
		}
		pOut[i*dim]=pOut[i*dim+1];
	}

}

void rmNoiseCC(float* hboxmap, int box, int nframe, int peakR)
{
	int peakD=peakR*2-1;
	float *peak=new float[peakD*peakD];
	float *buf=0;
	memset(peak,0,sizeof(float)*peakD*peakD);

	int i,j,k;
	int offset=box/2-peakR+1;
	for(k=0;k<nframe;k++)
	{
		buf=hboxmap+box*box*k;
		for(j=0;j<peakD;j++)
			for(i=0;i<peakD;i++)
			{
				peak[j*peakD+i]+=buf[(j+offset)*box+i+offset]/nframe;
			}
	}

	for(k=0;k<nframe;k++)
	{
		buf=hboxmap+box*box*k;
		for(j=0;j<peakD;j++)
			for(i=0;i<peakD;i++)
			{
				buf[(j+offset)*box+i+offset]-=peak[j*peakD+i];
			}
	}

	delete [] peak;

}

void cosmask2d(complex<float> *pfft, int nsam)
{
	int i,j,ii,jj,id;
	
	int hnsam=nsam/2;
	int hnsamb=hnsam+1;
	float step=3.141592653589793/hnsam;
	int r;
	for(j=0;j<hnsam;j++)
		for(i=0;i<hnsamb;i++)
		{
			id=j*hnsamb+i;
			r=int(sqrt(float(i*i+j*j))+0.5);
			if(r<hnsam)	pfft[id]*=cos(r*step)/2+0.5;
			else pfft[id]=0;

			id=(j+hnsam)*hnsamb+i;
			jj=j-hnsam;
			r=int(sqrt(float(i*i+jj*jj))+0.5);
			if(r<hnsam)	pfft[id]*=cos(r*step)/2+0.5;
			else pfft[id]=0;
		}
}

void histMinMax(float *buf, int size, float &min, float &max, float threshold)
{
	int nbin=400;
	int *hist=new int[nbin];
	memset(hist,0,sizeof(int)*nbin);

	//find real min and max
	min=1e20;
	max=-1e20;
	int i,id;
	for(i=0;i<size;i++)
	{
		if(buf[i]<min) min=buf[i];
		if(buf[i]>max) max=buf[i];
	}
	if(min>=max) return;

	//get histogram
	float step=(max-min)/nbin;
	for(i=0;i<size;i++)
	{
		id=int((buf[i]-min)/step);
		if(id>=nbin) id=nbin-1;
		hist[id]++;
	}

	//remove 1% 
	int percent=int(size*threshold);
	int count=0;
	for(i=0;i<nbin;i++)
	{
		count+=hist[i];
		if(count>percent)
		{
			min+=i*step;
			break;
		}
	}
	count=0;
	for(i=nbin-1;i>0;i--)
	{
		count+=hist[i];
		if(count>percent)
		{
			max-=(nbin-1-i)*step;
			break;
		}
	}


	delete [] hist;
}

void buf2Disp(char *disp, float *buf, int size)
{
	float min,max;
	int i;
	histMinMax(buf,size, min, max);
	if(min>=max) return;

	float scale=255.0/(max-min);
	for(i=0;i<size;i++)
	{
		if(buf[i]<=min) disp[i]=0;
		else if(buf[i]>=max) disp[i]=255;
		else disp[i]=char((buf[i]-min)*scale);
	}
}


void buf2DispShort(short *disp, float *buf, int size)
{
	float min,max;
	int i;
	histMinMax(buf,size, min, max);
	if(min>=max) return;

	float scale=255.0/(max-min);
	for(i=0;i<size;i++)
	{
		if(buf[i]<=min) disp[i]=0;
		else if(buf[i]>=max) disp[i]=255;
		else disp[i]=short((buf[i]-min)*scale);
	}
}

//sizeof(pIn)=(nsam/2+1)*nsam;
//sizeof(pOut)=(nsam+2)*nsam;
void FFTModulusToDispBuf(float *pIn, float *pOut, DIM nsam)
{
	int hnsam=nsam.width()/2;
	int hnsamb=nsam.width()/2+1;
	int nsamb=nsam.width()+2;

	int i,j,ii,jj,id0,id1;
	//bottom-right
	for(i=0;i<hnsam;i++)
	{
		memcpy(pOut+(hnsam+i)*nsamb+hnsam,pIn+i*hnsamb,sizeof(float)*hnsamb);
	}
	//top-right
	for(i=0;i<hnsam;i++)
	{
		memcpy(pOut+i*nsamb+hnsam,pIn+(nsam.height()/2+i)*hnsamb,sizeof(float)*hnsamb);
	}

	//get another half
	for(j=1;j<nsam.height();j++)
	{
		id0=j*nsamb;
		id1=(nsam.height()-j)*nsamb;
		for(i=0;i<hnsam;i++)
		{
			pOut[id0+i]=pOut[id1+nsam.width()-i];
		}
	}
	//set first half row which doesn't have corresping line
	memcpy(pOut,pOut+nsamb,sizeof(float)*hnsam);
}

void BinFFTDispBufToChar(char *pDisp, int dispdim, float *pfft, DIM nsam)
{
	int i;
	int hdispdim=dispdim/2;
	int dispdimb=dispdim+2;
	int nsamb=nsam.width()+2;
	int dispsize=dispdim*dispdim;

	//fft
	fft2d_safe(pfft,nsam);
	//bin by FFT
	float *buf=new float[(dispdim+2)*dispdim];
	float *disp=new float[dispdim*dispdim];
	for(i=0;i<hdispdim;i++)
	{
		memcpy(buf+i*dispdimb,pfft+i*nsamb,sizeof(float)*dispdimb);
		memcpy(buf+(dispdim-1-i)*dispdimb,pfft+(nsam.height()-1-i)*nsamb,sizeof(float)*dispdimb);
	}

	//ifft disp
	ifft2d_safe(buf,DIM(dispdim,dispdim));

	//convert to char display
	fft2buf(disp,buf,DIM(dispdim,dispdim));
	for(i=0;i<dispsize;i++) disp[i]/=dispsize;
	buf2Disp(pDisp,disp,dispdim*dispdim);
	

	delete [] buf;
	delete [] disp;
}

void BinFFTDispBufToChar(short *pDisp, int dispdim, float *pfft, DIM nsam)
{
	int i;
	int hdispdim=dispdim/2;
	int dispdimb=dispdim+2;
	int nsamb=nsam.width()+2;
	int dispsize=dispdim*dispdim;

	//fft
	fft2d_safe(pfft,nsam);

	//bin by FFT
	float *buf=new float[(dispdim+2)*dispdim];
	float *disp=new float[dispdim*dispdim];
	for(i=0;i<hdispdim;i++)
	{
		memcpy(buf+i*dispdimb,pfft+i*nsamb,sizeof(float)*dispdimb);
		memcpy(buf+(dispdim-1-i)*dispdimb,pfft+(nsam.height()-1-i)*nsamb,sizeof(float)*dispdimb);
	}

	//ifft disp
	ifft2d_safe(buf,DIM(dispdim,dispdim));

	//convert to char display
	fft2buf(disp,buf,DIM(dispdim,dispdim));
	for(i=0;i<dispsize;i++) disp[i]/=dispsize;
	buf2DispShort(pDisp,disp,dispdim*dispdim);
	

	delete [] buf;
	delete [] disp;
}

void DispFFTToDispShort(short *pDisp, float *pfft, int dispdim)
{
	int i;
	int dispsize=dispdim*dispdim;

	//ifft disp
	ifft2d_safe(pfft,DIM(dispdim,dispdim));

	//convert to char display
	float *disp=new float[dispdim*dispdim];
	fft2buf(disp,pfft,DIM(dispdim,dispdim));
	for(i=0;i<dispsize;i++) disp[i]/=dispsize;
	buf2DispShort(pDisp,disp,dispdim*dispdim);
	
	delete [] disp;
}

float* ReadRef(const char *filename, int dst_nx, int dst_ny)
{
	MRC mrc;
	if(mrc.open(filename,"rb")<=0) return 0;
	
	int nx=mrc.getNx();
	int ny=mrc.getNy();
	
	if(nx!=dst_nx || ny!=dst_ny) 
	{
		mrc.close();
		return 0;
	}
	
	int size=nx*ny;
	float *ref=new float[size];
	
	if(mrc.read2DIm_32bit(ref,0)!=mrc.getImSize())
	{
		mrc.close();
		return 0;
	}
	
	mrc.close();
	
	return ref;

}

bool ApplyRef(float* im, bool bDark, float* dark, bool bGain, float* gain, size_t size)
{
	size_t i;
	if(im==0) return false;
	if(!bDark && !bGain) return true;
	
	
	if(bDark && dark==0) return false;
	if(bGain && gain==0) return false;
	
	if(bDark && !bGain) for(i=0;i<size;i++) im[i]-=dark[i];
	if(!bDark && bGain) for(i=0;i<size;i++) im[i]*=gain[i];
	if(bDark && bGain) for(i=0;i<size;i++) im[i]=(im[i]-dark[i])*gain[i];
	
	return true;
}





