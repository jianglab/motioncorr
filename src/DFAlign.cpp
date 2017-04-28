#include "DFAlign.h"
#include "func.h"
#include "mfunc.h"
#include "safefft.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>


CDFAlign::CDFAlign(void)
{
	m_nsam=0;
	//m_nsamRaw=0;
	//For image output
	m_bufIm=new float[(DISPDIM+2)*DISPDIM];
	m_dispIm=new short[DISPDIM*DISPDIM];

	m_bufFFTCorr=0;
	m_dispFFTCorr=new short[DISPDIM*DISPDIM];

	m_bufFFTRaw=0;
	m_dispFFTRaw=new short[DISPDIM*DISPDIM];

	m_bufCCMap=0;

	m_pGain=0;
	m_pDark=0;

	strcpy(m_fnGain, "none");
	strcpy(m_fnDark, "none");

	initFFTWLock();

}


CDFAlign::~CDFAlign(void)
{
	if(m_bufIm!=0) delete [] m_bufIm;
	if(m_dispIm!=0) delete [] m_dispIm;

	if(m_bufFFTCorr!=0) delete [] m_bufFFTCorr;
	if(m_dispFFTCorr!=0) delete [] m_dispFFTCorr;

	if(m_bufFFTRaw!=0) delete [] m_bufFFTRaw;
	if(m_dispFFTRaw!=0) delete [] m_dispFFTRaw;

	if(m_bufCCMap!=0) delete [] m_bufCCMap;

	if(m_pGain!=0) delete [] m_pGain;
	if(m_pDark!=0) delete [] m_pDark;

	freeFFTWLock();
}


void CDFAlign::Message(const char *str)
{
	printf("%s\n",str);
}

void CDFAlign::UpdateDisplay()
{}

void CDFAlign::TextOutput(const char *str)
{
	m_log=str;
	printf("%s",m_log.c_str());
	//SendMessage(m_dlgwnd, WM_TSHOWLOG, 0,0);

	if(m_para.bSaveLog)
	{
		FILE *fp=fopen(m_fnLog,"a");
		fprintf(fp,"%s",m_log.c_str());
		fclose(fp);
	}
}

void* CDFAlign::ImageOutputThread(void *p)
{
	CDFAlign *pThis=(CDFAlign *)p;

	DIM dispdim=pThis->m_dispdim;

	ifft2d(pThis->m_bufIm,dispdim);

	int size=dispdim.width()*dispdim.height();
	float *buf=new float[size];
	fft2buf(buf,pThis->m_bufIm,dispdim);
	buf2DispShort(pThis->m_dispIm, buf, size);

	//add output code here
	//SendMessage(pThis->m_dlgwnd, WM_TSHOWIMAGE, 0,0);
	MRC mrc;
	mrc.open(pThis->m_dispCorrSum,"wb");
	mrc.createMRC(pThis->m_dispIm,dispdim.width(),dispdim.height(),1);
	mrc.close();

	delete [] buf;

	return (void *)0;
}
void CDFAlign::ImageOutput(float *buf)
{
	DIM dispdim=m_dispdim;
	memcpy(m_bufIm,buf,sizeof(float)*(dispdim.width()+2)*dispdim.height());

	pthread_t tid;
	int terror;
	terror=pthread_create(&tid,NULL,ImageOutputThread,(void *)this);
  	if(terror!=0)
  	{
		TextOutput("Error: Failed to create pthread: Image Output\n");;
   	return;
   }
   m_tids.push_back(tid);

}

void* CDFAlign::FFTOutputCorrThread(void *p)
{
	CDFAlign *pThis=(CDFAlign *)p;
	//DIM nsam=pThis->m_nsam.MinSquare();
	//float *buf=new float[(nsam.width()+2)*nsam.height()];
	//FFTModulusToDispBuf(pThis->m_bufFFTCorr, buf, nsam);
	//BinFFTDispBufToChar(pThis->m_dispFFTCorr, DISPDIM, buf, nsam);
	//delete [] buf;
	DispFFTToDispShort(pThis->m_dispFFTCorr,pThis->m_bufFFTCorr,DISPDIM);

	//add output code here
	//SendMessage(pThis->m_dlgwnd, WM_TSHOWFFTCORR, 0,0);
	MRC mrc;
	mrc.open(pThis->m_dispCorrFFT,"wb");
	mrc.createMRC(pThis->m_dispFFTCorr,DISPDIM,DISPDIM,1);
	mrc.close();

	return (void *)0;
}
void CDFAlign::FFTOutputCorr(float *buf)
{
	//DIM nsam=m_nsam.MinSquare();
	//if(m_bufFFTCorr==0) m_bufFFTCorr=new float[(nsam.width()/2+1)*nsam.height()];
	//memcpy(m_bufFFTCorr,buf,sizeof(float)*(nsam.width()/2+1)*nsam.height());
	if(m_bufFFTCorr==0) m_bufFFTCorr=new float[(DISPDIM+2)*DISPDIM];
	memcpy(m_bufFFTCorr,buf,sizeof(float)*(DISPDIM+2)*DISPDIM);

	pthread_t tid;
	int terror;
	terror=pthread_create(&tid,NULL,FFTOutputCorrThread,(void *)this);
  	if(terror!=0)
  	{
		TextOutput("Error: Failed to create pthread: FFT Output Corr\n");;
   	return;
   }
   m_tids.push_back(tid);
}

void* CDFAlign::FFTOutputRawThread(void *p)
{
	CDFAlign *pThis=(CDFAlign *)p;
	//DIM nsam=pThis->m_nsam.MinSquare();
	//float *buf=new float[(nsam.width()+2)*nsam.height()];
	//FFTModulusToDispBuf(pThis->m_bufFFTRaw, buf, nsam);
	//BinFFTDispBufToChar(pThis->m_dispFFTRaw, DISPDIM, buf, nsam);
	//delete [] buf;
	DispFFTToDispShort(pThis->m_dispFFTRaw,pThis->m_bufFFTRaw,DISPDIM);

	//add output code here
	//SendMessage(pThis->m_dlgwnd, WM_TSHOWFFTRAW, 0,0);
	MRC mrc;
	mrc.open(pThis->m_dispRawFFT,"wb");
	mrc.createMRC(pThis->m_dispFFTRaw,DISPDIM,DISPDIM,1);
	mrc.close();



	return (void *)0;
}
void CDFAlign::FFTOutputRaw(float *buf)
{
	//DIM nsam=m_nsam.MinSquare();
	//if(m_bufFFTRaw==0) m_bufFFTRaw=new float[(nsam.width()/2+1)*nsam.height()];
	//memcpy(m_bufFFTRaw,buf,sizeof(float)*(nsam.width()/2+1)*nsam.height());
	if(m_bufFFTRaw==0) m_bufFFTRaw=new float[(DISPDIM+2)*DISPDIM];
	memcpy(m_bufFFTRaw,buf,sizeof(float)*(DISPDIM+2)*DISPDIM);

	pthread_t tid;
	int terror;
	terror=pthread_create(&tid,NULL,FFTOutputRawThread,(void *)this);
  	if(terror!=0)
  	{
		TextOutput("Error: Failed to create pthread: FFT Output Raw\n");;
   	return;
   }
   m_tids.push_back(tid);
}

void* CDFAlign::CCMapOutputThread(void *p)
{
	CDFAlign *pThis=(CDFAlign *)p;

	//add output code here
	//SendMessage(pThis->m_dlgwnd, WM_TSHOWCCMAP, 0,0);

	return (void *)0;
}
void CDFAlign::CCMapOutput(float *buf, void *pki)
{
	Vector<double> &ki=*(Vector<double> *)pki;
	if(m_bufCCMap!=0) delete [] m_bufCCMap;
	int size=m_para.CCPeakSearchDim*m_para.CCPeakSearchDim*ki.size();
	m_bufCCMap=new float[size];
	memcpy(m_bufCCMap,buf,sizeof(float)*size);
	m_kiCCMap.clear();
	for(int i=0;i<ki.size();i++) m_kiCCMap.push_back(ki[i]);

	pthread_t tid;
	int terror;
	terror=pthread_create(&tid,NULL,CCMapOutputThread,(void *)this);
  	if(terror!=0)
  	{
		TextOutput("Error: Failed to create pthread: CC Map Output\n");;
   	return;
   }
   m_tids.push_back(tid);

}

void CDFAlign::PlotFSC(float2* hRaw0, float2 *hRaw1, float2 *hCorr0, float2 *hCorr1,
					MASK *pPosList, DIM nsam, complex<double> direction)
{
	const int step=nsam.width()/200;
	float ratio = float(nsam.x)/float(nsam.y);
	float ratio2 = ratio*ratio;

	int nsamc=nsam.width()/2+1;
	int sizec=nsamc*nsam.height();
	int nbox=nsamc/step+1;

	int i,id;
	float r,angle;
	float edge=cos(PI/4);

	float3 *fRaw=new float3[nbox];   // x:cos(phase) y:amp^2 z:amp^2, overall
	float3 *fRaw0=new float3[nbox];   // x:cos(phase) y:amp^2 z:amp^2, along drift
	float3 *fRaw1=new float3[nbox];   // x:cos(phase) y:amp^2 z:amp^2, perpendicular to drift
	float3 *fCorr=new float3[nbox];   // x:cos(phase) y:amp^2 z:amp^2, overall
	float3 *fCorr0=new float3[nbox];   // x:cos(phase) y:amp^2 z:amp^2, along drift
	float3 *fCorr1=new float3[nbox];   // x:cos(phase) y:amp^2 z:amp^2, perpendicular to drift
	memset(fRaw,0,sizeof(float3)*nbox);
	memset(fRaw0,0,sizeof(float3)*nbox);
	memset(fRaw1,0,sizeof(float3)*nbox);
	memset(fCorr,0,sizeof(float3)*nbox);
	memset(fCorr0,0,sizeof(float3)*nbox);
	memset(fCorr1,0,sizeof(float3)*nbox);

	if(abs(direction)>0.1) direction/=abs(direction);
	else direction=1.0;

	cuComplex a,b;


	for(i=0;i<sizec;i++)
	{
		r=sqrt(float(pPosList[i].x*pPosList[i].x+pPosList[i].y*pPosList[i].y*ratio2));
		if(int(r)<=0 || r>=nsamc) continue;
		id=int(r/step);

		angle=fabs(pPosList[i].x*direction.real()+pPosList[i].y*ratio*direction.imag())/r;

		if(angle>edge) //along drift
		{
			a=hRaw0[i];
			b=hRaw1[i];
			fRaw0[id].x+=a.x*b.x+a.y*b.y;
			fRaw0[id].y+=a.x*a.x+a.y*a.y;
			fRaw0[id].z+=b.x*b.x+b.y*b.y;

			a=hCorr0[i];
			b=hCorr1[i];
			fCorr0[id].x+=a.x*b.x+a.y*b.y;
			fCorr0[id].y+=a.x*a.x+a.y*a.y;
			fCorr0[id].z+=b.x*b.x+b.y*b.y;
}
		else  //perpendicular drift
		{
			a=hRaw0[i];
			b=hRaw1[i];
			fRaw1[id].x+=a.x*b.x+a.y*b.y;
			fRaw1[id].y+=a.x*a.x+a.y*a.y;
			fRaw1[id].z+=b.x*b.x+b.y*b.y;

			a=hCorr0[i];
			b=hCorr1[i];
			fCorr1[id].x+=a.x*b.x+a.y*b.y;
			fCorr1[id].y+=a.x*a.x+a.y*a.y;
			fCorr1[id].z+=b.x*b.x+b.y*b.y;
		}

		//Overall
		a=hRaw0[i];
		b=hRaw1[i];
		fRaw[id].x+=a.x*b.x+a.y*b.y;
		fRaw[id].y+=a.x*a.x+a.y*a.y;
		fRaw[id].z+=b.x*b.x+b.y*b.y;
		a=hCorr0[i];
		b=hCorr1[i];
		fCorr[id].x+=a.x*b.x+a.y*b.y;
		fCorr[id].y+=a.x*a.x+a.y*a.y;
		fCorr[id].z+=b.x*b.x+b.y*b.y;
	}

	m_fscRaw.resize(nbox,0.0);
	m_fscRaw0.resize(nbox,0.0);
	m_fscRaw1.resize(nbox,0.0);
	m_fscCorr.resize(nbox,0.0);
	m_fscCorr0.resize(nbox,0.0);
	m_fscCorr1.resize(nbox,0.0);


	float t;
	for(i=0;i<nbox;i++)
	{
		//Raw overall
		t=fRaw[i].y*fRaw[i].z;
		if(t>0.00001) m_fscRaw[i]=complex<double>(double(i)/nbox,fRaw[i].x/sqrt(t));
		else m_fscRaw[i]=complex<double>(double(i)/nbox,0.0);

		//Raw along drift
		t=fRaw0[i].y*fRaw0[i].z;
		if(t>0.00001) m_fscRaw0[i]=complex<double>(double(i)/nbox,fRaw0[i].x/sqrt(t));
		else m_fscRaw0[i]=complex<double>(double(i)/nbox,0.0);

		//Raw perpendicualr to drift
		t=fRaw1[i].y*fRaw1[i].z;
		if(t>0.00001) m_fscRaw1[i]=complex<double>(double(i)/nbox,fRaw1[i].x/sqrt(t));
		else m_fscRaw1[i]=complex<double>(double(i)/nbox,0.0);

		//Corrected overall
		t=fCorr[i].y*fCorr[i].z;
		if(t>0.00001) m_fscCorr[i]=complex<double>(double(i)/nbox,fCorr[i].x/sqrt(t));
		else m_fscCorr[i]=complex<double>(double(i)/nbox,0.0);

		//Corrected along drift
		t=fCorr0[i].y*fCorr0[i].z;
		if(t>0.00001) m_fscCorr0[i]=complex<double>(double(i)/nbox,fCorr0[i].x/sqrt(t));
		else m_fscCorr0[i]=complex<double>(double(i)/nbox,0.0);

		//Corrected perpendicualr to drift
		t=fCorr1[i].y*fCorr1[i].z;
		if(t>0.00001) m_fscCorr1[i]=complex<double>(double(i)/nbox,fCorr1[i].x/sqrt(t));
		else m_fscCorr1[i]=complex<double>(double(i)/nbox,0.0);
	}

	//add output code here
	//SendMessage(m_dlgwnd, WM_TSHOWFSC, 0,0);

	//output

	char str[512];
	sprintf(str,"\nFSC: Overall(O), parallel(D) and perpendicular(U) to drift direction(%.1f Deg):\n",
				atan2(direction.imag(),-direction.real())*180.0/PI);
	TextOutput(str);
	TextOutput("   Nq%     O_Raw     D_Raw     U_Raw     O_Corr    D_Corr    U_Corr\n");
	for(i=0;i<nbox;i++)
	{
		sprintf(str,"%7.2f %9.4f %9.4f %9.4f %9.4f %9.4f %9.4f\n",i*100.0/nbox,
			m_fscRaw[i].imag(),m_fscRaw0[i].imag(),m_fscRaw1[i].imag(),
			m_fscCorr[i].imag(),m_fscCorr0[i].imag(),m_fscCorr1[i].imag());
		TextOutput(str);
	}
	TextOutput("\n");

	delete [] fRaw;
	delete [] fRaw0;
	delete [] fRaw1;
	delete [] fCorr;
	delete [] fCorr0;
	delete [] fCorr1;
}

void CDFAlign::Done()
{

	//add output code here
	//SendMessage(m_dlgwnd, WM_TDONE, 0,0);

}

void CDFAlign::PlotOutput(vector<complex<double> > &xy)
{
	m_curve=xy;
	//add output code here
	//SendMessage(m_dlgwnd, WM_TSHOWCURVE, 0,0);

}

int CDFAlign::getNFrame()
{
	MRC mrc;
	if(mrc.open(m_fnStack,"rb")<=0) return 0;
	int n=mrc.getNz();
	mrc.close();
	return n;
}
int CDFAlign::getNFrame(const char *filename)
{
	MRC mrc;
	if(mrc.open(filename,"rb")<=0) return 0;
	int n=mrc.getNz();
	mrc.close();
	return n;
}

MRCHeader CDFAlign::getMrcHeader(const char *filename)
{
	MRC mrc;
	MRCHeader header;
	memset(&header,0,sizeof(MRCHeader));
	if(mrc.open(filename,"rb")<=0) return header;
	mrc.getHeader(&header);;
	mrc.close();
	return header;
}

void CDFAlign::RunAlign()
{
	// TODO: Add your control notification handler code here
	//UpdateData(true);

	m_tids.clear();

	pthread_t tid;
	int terror;
	terror=pthread_create(&tid,NULL,ThreadFunc_cuAlign,(void *)this);
  	if(terror!=0)
  	{
		TextOutput("Error: Failed to create pthread: Align\n");
		return;
   }
   m_tids.push_back(tid);

   //wait for finish
   void *TReturn;
   int i;
	for(i=0;i<m_tids.size();i++)
	{
   	terror=pthread_join(m_tids[i],&TReturn);
   	if(terror!=0)
   	{
      	TextOutput("Warning: Thread doesn't exit. Something may be wrong.\n");
   	}

   }
   m_tids.clear();
}


void* CDFAlign::ThreadFunc_cuAlign(void* p)
{
	CDFAlign *pThis=(CDFAlign *)p;
	APARA &para=pThis->m_para;
	pThis->m_bRun=true;

	char str[512];
	int i,j;
	bool bSuccess=false;

	//open stack file
	MRC stack;
	if(stack.open(pThis->m_fnStack,"rb")<=0)
	{
		sprintf(str,"Error: Failed to open stack %s .",pThis->m_fnStack);
		Message(str);
		pThis->m_bRun=false;
		return (void *)0;
	}

	//get image size
	int nx=stack.getNx();
	int ny=stack.getNy();
	int nz=stack.getNz();
	sprintf(str,"\nInput Stack: Nx(%d) Ny(%d) Nz(%d)\n\n",nx,ny,nz);
	pThis->TextOutput(str);


	int bin=para.bin;
	if(bin<=0) return (void *)0;

	int offsetx=para.crop_offsetx;
	int offsety=para.crop_offsety;
	DIM nsamUnbin=verifyCropSize(nx,ny, offsetx, offsety, para.crop_nsam,bin);
	if(nsamUnbin.x<=0 || nsamUnbin.y<=0 )
	{
		Message("Error: Wrong image Size.");
		pThis->m_bRun=false;
		return (void *)0;
	}

	DIM nsam=nsamUnbin/bin;

	int nsamb=nsam.width()+2;
	if(bin==1) sprintf(str,"Crop Image: Offset(%d %d) Dim(%d %d)\n\n",
							offsetx,offsety,nsamUnbin.x,nsamUnbin.y);
	else sprintf(str,"Crop Image: Offset(%d %d) RawDim(%d %d) BinnedDim(%d %d)\n\n",
							offsetx,offsety,nsamUnbin.x,nsamUnbin.y,nsam.x,nsam.y);
	pThis->TextOutput(str);
	pThis->m_nsam=nsam;
	//pThis->m_nsamRaw=nx/bin;

	//allocate memeory
	size_t size=nsam.width()*nsam.height();
	size_t sizeb=nsamb*nsam.height();
	int sizebUnbin=(nsamUnbin.width()+2)*nsamUnbin.height();
	if(para.nStart<0) para.nStart=0;
	if(para.nEnd>=nz) para.nEnd=nz-1;
	int nframe=para.nEnd-para.nStart+1;
	pThis->UpdateDisplay();

	//host memory
	float *bufmrc=new float[nx*ny];
	float *bufmrcfft=new float[sizebUnbin];
	float *htmp=new float[sizeb];
	float *hbuf=new float[sizeb*nframe];  //host memory for entir stack
	float *hdisp=new float[sizeb];
	float *hsumRaw=new float[sizeb];
	float *hsumCorr=new float[sizeb];
	float *hFSCRaw0=new float[sizeb];  //even number
	float *hFSCRaw1=new float[sizeb];  //odd
	float *hFSCCorr0=new float[sizeb];  //even number
	float *hFSCCorr1=new float[sizeb];   //odd

	size_t refsize=0;
	//if both -hgr and -fgr are enabled, -fgr will be disabled.
	if(para.bHGain && stack.getMode()==5)
	{
		if(pThis->m_pGain!=0) delete [] pThis->m_pGain;
		pThis->m_pGain=new float[nx*ny];
		if( stack.readGainInHeader(pThis->m_pGain) == (nx*ny*sizeof(float)) )
		{
			refsize+=nx*ny;
			para.bGain=true;
			sprintf(str,"\nGain reference was read from MRC header(Mode=5)\n\n");
			pThis->TextOutput(str);
		}
		else
		{
			Message("\nFailed to read gain reference from MRC header(Mode=5).");
			delete [] pThis->m_pGain;
			pThis->m_pGain=0;

		}
	}
	else if(para.bGain)
	{
		if(pThis->m_pGain!=0) delete [] pThis->m_pGain;
		pThis->m_pGain=ReadRef(pThis->m_fnGain,nx,ny);
		if(pThis->m_pGain==0)
		{
			sprintf(str,"\nFailed to read gain reference from %s .\n\n",pThis->m_fnGain);
			para.bGain=false;
		}
		else
		{
			sprintf(str,"\nGain reference was read from %s .\n\n",pThis->m_fnGain);
			pThis->TextOutput(str);
		refsize+=nx*ny;
		}
	}

	if(para.bDark)
	{
		if(pThis->m_pDark!=0) delete [] pThis->m_pDark;
		pThis->m_pDark=ReadRef(pThis->m_fnDark,nx,ny);
		if(pThis->m_pDark==0)
		{
			sprintf(str,"\nFailed to read dark reference from %s .\n\n",pThis->m_fnDark);
			para.bDark=false;
		}
		else
		{
			sprintf(str,"\nDark reference was read from %s .\n\n",pThis->m_fnDark);
			pThis->TextOutput(str);
			refsize+=nx*ny;
		}
	}




	sprintf(str,"Allocate host memory: %f Gb\n",(nx*ny+sizeb*(nframe+8)+sizebUnbin+refsize)/256.0/1024.0/1024.0);
	pThis->TextOutput(str);
	if(hbuf==0)
	{
		if(bufmrc!=NULL) delete [] bufmrc;
		Message("Failed to allocate host memeory.");
		pThis->m_bRun=false;
		return (void *)0;
	}


	//device memory
	bool success=initGPU(para.GPUNum);
	if(!success)
	{
		sprintf(str,"Failed to initialize GPU #%d.",para.GPUNum);
		Message(str);
		delete [] bufmrc;
		delete [] hbuf;
		pThis->m_bRun=false;
		return (void *)0;
	}


	float *dsum=0;
	float *dsumcorr=0;
	float *dfft=0;
	float *dtmp=0;
	GPUMemAlloc((void **)&dsum,sizeof(float)*sizeb);
	GPUMemAlloc((void **)&dsumcorr,sizeof(float)*sizeb);
	GPUMemAlloc((void **)&dtmp,sizeof(float)*sizeb);
	cufftHandle fft_plan,ifft_plan;

	//prepare fft for unbinned image
	fft_plan=GPUFFTPlan(nsamUnbin);
	GPUSync();
	GPUMemAlloc((void **)&dfft,sizeof(float)*sizebUnbin);


	//make a list
	int sizec=(nsam.width()/2+1)*nsam.height();
	MASK *hPosList=new MASK[sizec];
	MASK *dPosList=0;
	MkPosList(hPosList,nsam,para.bfactor);
	GPUMemAlloc((void **)&dPosList,sizeof(MASK)*sizec);
	GPUMemH2D((void **)dPosList,(void **)hPosList,sizeof(MASK)*sizec);

	size_t theFree, theTotal;
	GPUMemCheck(theFree,theTotal);
	sprintf(str,"GPU memory:  free:%.0fMb    total:%.0fMb\n", theFree/1024.0/1024.0, theTotal/1024.0/1024.0);
	pThis->TextOutput(str);

	//Read stack
	pThis->TextOutput("\nRead stack:\n");

	float sx=0;
	float sy=0;
	float shiftx,shifty,cc;
	float avgcc=0.0;
	bool bFSCEven=true;

	//prepare frame sum numbers
	//Only the frame inside sum range is used for stack and sum output
	int nStartSum=para.nStartSum-para.nStart;
	int nEndSum=para.nEndSum-para.nStart;
	if(nStartSum<0) nStartSum=0;
	if(para.nEndSum>para.nEnd) nEndSum=para.nEnd-para.nStart+1;
	if(nEndSum<=nStartSum || nEndSum>=nframe) nEndSum=nframe-1;

	//1. calculate sum
	GPUMemZero((void **)&dsum,sizeof(float)*sizeb);
	GPUSync();
	GPUMemZero((void **)&dsumcorr,sizeof(float)*sizeb);
	GPUSync();
	for(j=para.nStart;j<=para.nEnd;j++)
	{
		//read from file and crop
		if(stack.read2DIm_32bit(bufmrc,j)!=stack.getImSize())
		{
			sprintf(str,"Error when reading #%03d\n",j);
			pThis->TextOutput(str);
		}

		if(para.flp) {	//flip image along y axis
			FlipYAxis(bufmrc, nx, ny);
		}

		//apply gain and dark reference
		if(!ApplyRef(bufmrc, para.bDark, pThis->m_pDark, para.bGain, pThis->m_pGain, nx*ny))
		{
			sprintf(str,"Error when applying dark and/or gain to #%03d\n",j);
			pThis->TextOutput(str);
		}


		crop2fft(bufmrc,DIM(nx,ny),bufmrcfft,offsetx,offsety,nsamUnbin,bin);

		//copy to GPU
		GPUMemH2D((void *)dfft,(void *)bufmrcfft,sizeof(float)*sizebUnbin);
		//do fft
		GPUFFT2d(dfft,fft_plan);
		GPUSync();

		//do binning
		if(bin>1)
		{
			GPUMemBinD2D(dtmp, dfft, nsam, nsamUnbin);
			GPUMemD2D(dfft, dtmp, sizeof(float)*sizeb);
		}

		//Sum
		if( (j-para.nStart)>=nStartSum && (j-para.nStart)<=nEndSum )
		{
			if(bFSCEven) GPUAdd(dsum,dfft,sizeb);
			else GPUAdd(dsumcorr,dfft,sizeb);
			bFSCEven=!bFSCEven;
		}
		//copy ffted image to host
		GPUMemD2H((void *)(hbuf+(j-para.nStart)*sizeb),(void *)dfft,sizeof(float)*sizeb);
		GPUSync();

		sprintf(str,"......Read and sum frame #%03d   mean:%f\n",j,(hbuf+(j-para.nStart)*sizeb)[0]/nsam.x/nsam.y);
		pThis->TextOutput(str);

	}
	GPUMemD2H((void *)hFSCRaw0,(void *)dsum,sizeof(float)*sizeb);
	GPUMemD2H((void *)hFSCRaw1,(void *)dsumcorr,sizeof(float)*sizeb);
	GPUAdd(dsum,dsumcorr,sizeb);
	GPUSync();


	//free memory for unbined image
	delete [] bufmrcfft;
	bufmrcfft=0;
	GPUMemFree((void **)&dfft);
	GPUFFTDestroy(fft_plan);
	fft_plan=0;
	//finish GPU memory allocate
	GPUMemAlloc((void **)&dfft,sizeof(float)*sizeb);
	GPUMemZero((void **)&dsumcorr,sizeof(float)*sizeb);
	GPUSync();
	ifft_plan=GPUIFFTPlan(nsam);
	GPUSync();

	//copy sum image to host for save and display
	if(para.bDispFFTRaw || para.bSaveRawSum)
	{
		GPUIFFT2d(dsum,ifft_plan);
		GPUSync();
		GPUMultiplyNum(dsum,1.0/size,sizeb);
		GPUMemD2H((void *)hsumRaw,(void *)dsum,sizeof(float)*sizeb);
		fft2buf(bufmrc,hsumRaw,nsam);
	}

	//save
	MRC mrcraw;
	if(para.bSaveRawSum)
	{
		//write to file
		mrcraw.open(pThis->m_fnRawsum,"wb");
		mrcraw.createMRC(bufmrc,nsam.width(),nsam.height(),1);
		//stats
		sprintf(str,"Mean=%f   Min=%f   Max=%f\n",
					mrcraw.m_header.dmean,mrcraw.m_header.dmin,mrcraw.m_header.dmax);
		pThis->TextOutput(str);
		mrcraw.close();
		sprintf(str,"Save Uncorrected Sum to: %s\n",pThis->m_fnRawsum);
		pThis->TextOutput(str);
	}


	//save un-corrected stack
	MRC stackRaw;
	if(para.bSaveStackRaw)
	{
		pThis->TextOutput("\nWrite uncorrected stack:\n");
		stackRaw.open(pThis->m_fnStackRaw,"wb");
		stackRaw.m_header.nx=nsam.x;
		stackRaw.m_header.ny=nsam.y;
		stackRaw.m_header.nz=nEndSum-nStartSum+1;
		stackRaw.updateHeader();

		for(j=nStartSum;j<=nEndSum;j++)
		{
			//copy to GPU
			GPUMemH2D((void *)dfft,(void *)(hbuf+j*sizeb),sizeof(float)*sizeb);
			//ifft
			GPUIFFT2d(dfft,ifft_plan);
			GPUSync();
			GPUMultiplyNum(dfft,1.0/size,sizeb);
			GPUSync();
			GPUMemD2H((void *)htmp,(void *)dfft,sizeof(float)*sizeb);
			fft2buf(bufmrc,htmp,nsam);
			stackRaw.write2DIm(bufmrc,j-nStartSum);

			sprintf(str,"......Write frame #%03d\n",j+para.nStart);
			pThis->TextOutput(str);
		}
		MinMaxMean(bufmrc,nsam.x*nsam.y, stackRaw.m_header.dmin, stackRaw.m_header.dmax, stackRaw.m_header.dmean);
		stackRaw.updateHeader();
		sprintf(str,"Mean=%f   Min=%f   Max=%f\n",
					stackRaw.m_header.dmean,stackRaw.m_header.dmin,stackRaw.m_header.dmax);
		pThis->TextOutput(str);
		stackRaw.close();
		sprintf(str,"Save Uncorrected Stack to: %s\n",pThis->m_fnStackRaw);
		pThis->TextOutput(str);
	}

	// added by Wen Jiang
	// generate running average frames for CC
	float *hbuf_orig=hbuf;
	float *hbuf_ra=0;
	if(para.nrw>1) {
		sprintf(str,"\nStart generating running average of %d frames\n",para.nrw);
		pThis->TextOutput(str);

		hbuf_ra = new float[sizeb*nframe];  //host memory for entire stack
		memcpy(hbuf_ra, hbuf, sizeof(float)*sizeb*nframe);	// hbuf stores FFT of all frames
		for(int fi = 0; fi<nframe; fi++) {
			sprintf(str,"......Generating runing average of frame %d\n",fi);
			pThis->TextOutput(str);
			for(int ri=0; ri<para.nrw; ri++) {
				int fi2 = fi - para.nrw/2 + ri;
				if(fi2<0 || fi2==fi || fi2>=nframe) continue;
				for(size_t pi=0; pi<sizeb; pi++)
					hbuf_ra[sizeb*fi+pi]+=hbuf[sizeb*fi2+pi];
			}
		}
		hbuf = hbuf_ra;
		sprintf(str,"Running average of %d frames are generated\n\n",para.nrw);
		pThis->TextOutput(str);
	}

	//2. frame to frame shift
	pThis->TextOutput("\nCalculate relative drift between frames\n");
	Matrix<complex<double> > A;
	vector<complex<int> > compList;
	int ncomp=OD_SetEquation_All(A,compList, nframe, para.FrameDistOffset);
	Vector<complex<double> > b=Vector<complex<double> >(ncomp);
	int box=para.CCPeakSearchDim;
	float *hboxmap=new float[box*box*ncomp];
	int par0,par1;
	for(j=0;j<ncomp;j++)
	{
		par0=compList[j].real();
		par1=compList[j].imag();
		//copy to GPU
		GPUMemH2D((void *)dsum,(void *)(hbuf+par0*sizeb),sizeof(float)*sizeb);
		GPUMemH2D((void *)dfft,(void *)(hbuf+par1*sizeb),sizeof(float)*sizeb);
		//shift and cc
		sx=0;
		sy=0;
		GPUShiftCC(dfft, dsum, dPosList,sx, sy, nsam);
		GPUSync();
		//do ifft
		GPUIFFT2d(dfft,ifft_plan);
		GPUSync();
		//find shift
		cc=FindShift(dfft,nsam, hboxmap+j*box*box, box, shiftx, shifty, para.NoisePeakSize-1);
		b[j]=complex<double>(shiftx,shifty);
		avgcc+=cc;
		sprintf(str,"......%03d Frame #%03d VS #%03d xy-shift: %8.4f %8.4f      CC:%f\n",j,par0+para.nStart,par1+para.nStart,shiftx,shifty,cc);
		pThis->TextOutput(str);
	}

	// added by Wen Jiang
	// restore original stack buffer and delete running averages
	hbuf=hbuf_orig;
	if(hbuf_ra) delete[] hbuf_ra;

	//3. sovle overdetermined equation
	Vector<complex<double> > shift=lsSolver(A,b);
	Vector<double> ki=abs(A*shift-b);
	sprintf(str,"\n......ki: First round \n");
	pThis->TextOutput(str);
	for(j=0;j<ki.size();j++)
	{
		par0=compList[j].real();
		par1=compList[j].imag();
		sprintf(str,"......ki #%03d of Frame #%03d VS #%03d: %8.4lf \n",j+para.nStart,par0+para.nStart,par1+para.nStart,ki[j]);
		pThis->TextOutput(str);
	}
	sprintf(str,"................................Average ki: %8.4lf \n\n",sum(ki)/ki.size());
	pThis->TextOutput(str);
	//display CCMap
	if(para.bDispCCMap)
	{
		pThis->CCMapOutput(hboxmap,(void *)&ki);
	}
	//3.1 re-sovle overdetermined equation after removing large ki elments
	double kiThresh=para.kiThresh;
	vector<int> goodlist=OD_Threshold(A, b, ki, kiThresh);
	shift=lsSolver(A,b);
	ki=abs(A*shift-b);
	sprintf(str,"......ki: Second round \n");
	pThis->TextOutput(str);
	for(j=0;j<ki.size();j++)
	{
		par0=compList[goodlist[j] ].real();
		par1=compList[goodlist[j] ].imag();
		sprintf(str,"......ki #%03d of Frame #%03d VS #%03d: %8.4f \n",j+para.nStart,par0+para.nStart,par1+para.nStart,ki[j]);
		pThis->TextOutput(str);
	}
	sprintf(str,"................................Average ki: %8.4lf \n\n",sum(ki)/ki.size());
	pThis->TextOutput(str);

	//output final shift
	//calculate average shift
	double avgshift=0.0;
	for(j=0;j<shift.size();j++) avgshift+=abs(shift[j]);
	avgshift/=shift.size();
	sprintf(str,"Final shift (Average %8.4lf pixels/frame):\n",avgshift);
	//output
	pThis->TextOutput(str);
	vector<complex<double> > shiftlist;
	complex<double> totalshift=0;
	sprintf(str,"......Shift of Frame #%03d : %8.4f %8.4f\n",para.nStart,totalshift.real(),totalshift.imag());
	pThis->TextOutput(str);
	shiftlist.push_back(totalshift);
	for(j=0;j<shift.size();j++)
	{
		totalshift=totalshift+shift[j];
		sprintf(str,"......Shift of Frame #%03d : %8.4f %8.4f\n",j+para.nStart+1,totalshift.real(),totalshift.imag());
		pThis->TextOutput(str);
		shiftlist.push_back(totalshift);
	}
	pThis->PlotOutput(shiftlist);

	//save CCMap image
	if(para.bSaveCCmap)
	{
		buf2mrc(pThis->m_fnCCmap,hboxmap,box,box,ncomp);
		sprintf(str,"Save CC map to: %s\n",pThis->m_fnCCmap);
		pThis->TextOutput(str);
	}


	MRC stackCorr;
	if(para.bSaveStackCorr)
	{
		stackCorr.open(pThis->m_fnStackCorr,"wb");
		stackCorr.m_header.nx=nsam.x;
		stackCorr.m_header.ny=nsam.y;
		stackCorr.m_header.nz=nEndSum-nStartSum+1;
		stackCorr.updateHeader();
	}

	//3. correct xy-shift


	//reset memory
	GPUMemZero((void **)&dsum,sizeof(float)*sizeb);
	GPUSync();
	GPUMemZero((void **)&dsumcorr,sizeof(float)*sizeb);
	GPUSync();
	//calculate middle frame shift
	complex<double> midshift=0.0;
	int RefFrame=0;
	if(para.bAlignToMid==1) RefFrame=nz/2+1;
	if(para.bAlignToMid<=0) RefFrame=abs(para.bAlignToMid);
	if(para.bAlignToMid!=0)
	{
		if(RefFrame<para.nStart) RefFrame=para.nStart;
		if(RefFrame>para.nEnd) RefFrame=para.nEnd;
		if(para.nStartSum>para.nEnd) para.nStartSum=para.nEnd;
		for(j=0;j<RefFrame-para.nStart;j++) midshift+=shift[j];
	}
	sprintf(str,"\nSum Frame #%03d - #%03d (Reference Frame #%03d):\n",nStartSum+para.nStart,nEndSum+para.nStart,RefFrame);
	pThis->TextOutput(str);

	//Add(copy) first frame to GPU
	totalshift=0;
	for(j=1;j<nStartSum+1;j++)
	{
		totalshift+=shift[j-1];
	}
	GPUMemH2D((void *)dsumcorr,(void *)(hbuf+nStartSum*sizeb),sizeof(float)*sizeb);
	if(para.bAlignToMid) GPUShift(dsumcorr,dPosList,-totalshift.real()+midshift.real(),-totalshift.imag()+midshift.imag(), nsam);
	GPUSync();
	bFSCEven=false;
	sprintf(str,"......Add Frame #%03d with xy shift: %8.4lf %8.4lf\n",nStartSum+para.nStart,-totalshift.real()+midshift.real(),-totalshift.imag()+midshift.imag());
	pThis->TextOutput(str);
	//Save stack
	if(para.bSaveStackCorr)
	{
		GPUMemD2D((void *)dfft,(void *)dsumcorr,sizeof(float)*sizeb);
		GPUIFFT2d(dfft,ifft_plan);
		GPUSync();
		GPUMultiplyNum(dfft,1.0/size,sizeb);
		GPUSync();
		GPUMemD2H((void *)htmp,(void *)dfft,sizeof(float)*sizeb);
		fft2buf(bufmrc,htmp,nsam);
		stackCorr.write2DIm(bufmrc,0);
	}
	//*******
	//sum other frame
	for(j=nStartSum+1;j<=nEndSum;j++)
	{
		totalshift+=shift[j-1];

		//copy to GPU
		GPUMemH2D((void *)dfft,(void *)(hbuf+j*sizeb),sizeof(float)*sizeb);
		//shift
		GPUShift(dfft,dPosList,-totalshift.real()+midshift.real(),-totalshift.imag()+midshift.imag(), nsam);
		GPUSync();
		//Sum
		if(bFSCEven) GPUAdd(dsumcorr,dfft,sizeb);
		else GPUAdd(dsum,dfft,sizeb);
		bFSCEven=!bFSCEven;

		sprintf(str,"......Add Frame #%03d with xy shift: %8.4lf %8.4lf\n",j+para.nStart,-totalshift.real()+midshift.real(),-totalshift.imag()+midshift.imag());
		pThis->TextOutput(str);

		//save stack
		if(para.bSaveStackCorr)
		{
			GPUIFFT2d(dfft,ifft_plan);
			GPUSync();
			GPUMultiplyNum(dfft,1.0/size,sizeb);
			GPUSync();
			GPUMemD2H((void *)htmp,(void *)dfft,sizeof(float)*sizeb);
			fft2buf(bufmrc,htmp,nsam);
			stackCorr.write2DIm(bufmrc,j-nStartSum);
		}
	}
	if(para.bSaveStackCorr)
	{
		MinMaxMean(bufmrc,nsam.x*nsam.y,stackCorr.m_header.dmin, stackCorr.m_header.dmax, stackCorr.m_header.dmean);
	}
	//final sum
	GPUMemD2H((void *)hFSCCorr0,(void *)dsumcorr,sizeof(float)*sizeb);
	GPUMemD2H((void *)hFSCCorr1,(void *)dsum,sizeof(float)*sizeb);
	GPUAdd(dsumcorr,dsum,sizeb);
	GPUSync();

	//copy binned sum to display
	if(para.bDispSumCorr)
	{
		DIM dispdim(DISPDIM,DISPDIM);
		if(nsam.x<nsam.y) dispdim.x=int(DISPDIM*float(nsam.x)/float(nsam.y)+0.5)/2*2;
		if(nsam.x>nsam.y) dispdim.y=int(DISPDIM*float(nsam.y)/float(nsam.x)+0.5)/2*2;

		pThis->m_dispdim=dispdim;
		GPUMemBinD2H(hdisp, dsumcorr, dispdim, nsam);
		pThis->ImageOutput(hdisp);
	}

	//copy sum image to host
	float *tsum=dsumcorr;
	GPUIFFT2d(tsum,ifft_plan);
	GPUMultiplyNum(tsum,1.0/size,sizeb);
	GPUMemD2H((void *)hsumCorr,(void *)tsum,sizeof(float)*sizeb);
	fft2buf(bufmrc,hsumCorr,nsam);

	//save
	MRC mrc;
	mrc.open(pThis->m_fnAlignsum,"wb");
	mrc.createMRC(bufmrc,nsam.width(),nsam.height(),1);
	//stats
	sprintf(str,"Mean=%f   Min=%f   Max=%f\n",mrc.m_header.dmean,mrc.m_header.dmin,mrc.m_header.dmax);
	pThis->TextOutput(str);
	mrc.close();
	sprintf(str,"Save Corrected Sum to: %s\n",pThis->m_fnAlignsum);
	pThis->TextOutput(str);

	//close save Corrected stack
	if(para.bSaveStackCorr)
	{
		stackCorr.updateHeader();
		pThis->TextOutput("\nWrite corrected stack:\n");
		sprintf(str,"Mean=%f   Min=%f   Max=%f\n",
					stackCorr.m_header.dmean,stackCorr.m_header.dmin,stackCorr.m_header.dmax);
		pThis->TextOutput(str);
		stackCorr.close();
		sprintf(str,"Save Corrected Stack to: %s\n",pThis->m_fnStackCorr);
		pThis->TextOutput(str);
	}

	if(para.bLogFSC)
	{
		complex<double> avgshift=0.0;
		for(i=0;i<shift.size();i++)
		{
			avgshift+=shift[i]/abs(shift[i]);
		}
		pThis->PlotFSC((cuComplex *)hFSCRaw0, (cuComplex *)hFSCRaw1, (cuComplex *)hFSCCorr0,
					(cuComplex *)hFSCCorr1,hPosList,nsam,avgshift);
	}


	//free GPU FFT plan, new plan will be created for rectangular image
	//GPUFFTDestroy(fft_plan);
	GPUFFTDestroy(ifft_plan);


	///////////////////////////
	DIM nsamsub=nsam.MinSquare();
	//prepare new fft
	if(para.bDispFFTRaw || para.bDispFFTCorr) fft_plan=GPUFFTPlan(nsamsub);
	//Make Raw fft modulus for display
	if(para.bDispFFTRaw)
	{
		GPUMemH2D((void *)dsum,(void *)hsumRaw,sizeof(float)*sizeb);
		GPURectFFTLogModulus(dfft, dsum, dtmp, dsumcorr, nsam, para.fftscale,fft_plan);
		//copy to host, make pwr image
		GPUMemD2H(htmp,dfft,sizeof(float)*(nsamsub.width()/2+1)*nsamsub.height());
		FFTModulusToDispBuf(htmp,hdisp, nsamsub);
		//copy back to device
		GPUMemH2D(dtmp,hdisp,sizeof(float)*(nsamsub.width()+2)*nsamsub.height());
		//do binning
		GPUBinFFT(dfft, DISPDIM, dtmp, nsamsub, fft_plan);
		GPUMemD2H((void *)hdisp,(void *)dfft,sizeof(float)*(DISPDIM+2)*DISPDIM);
		//display
		pThis->FFTOutputRaw(hdisp);
	}
	//Make Corrected fft modulus for display
	if(para.bDispFFTCorr)
	{
		GPUMemH2D((void *)dsum,(void *)hsumCorr,sizeof(float)*sizeb);
		GPURectFFTLogModulus(dfft, dsum, dtmp, dsumcorr, nsam, para.fftscale,fft_plan);
		//copy to host, make pwr image
		GPUMemD2H(htmp,dfft,sizeof(float)*(nsamsub.width()/2+1)*nsamsub.height());
		FFTModulusToDispBuf(htmp,hdisp, nsamsub);
		//copy back to device
		GPUMemH2D(dtmp,hdisp,sizeof(float)*(nsamsub.width()+2)*nsamsub.height());
		//do binning
		GPUBinFFT(dfft, DISPDIM, dtmp, nsamsub, fft_plan);
		GPUMemD2H((void *)hdisp,(void *)dfft,sizeof(float)*(DISPDIM+2)*DISPDIM);
		//display
		pThis->FFTOutputCorr(hdisp);
	}
	//destory fft
	if(para.bDispFFTRaw || para.bDispFFTCorr) GPUFFTDestroy(fft_plan);
	/////////////////////////////////////////





	delete [] bufmrc;
	delete [] hbuf;
	delete [] hPosList;
	GPUMemFree((void **)&dPosList);
	GPUMemFree((void **)&dsum);
	GPUMemFree((void **)&dsumcorr);
	GPUMemFree((void **)&dfft);
	GPUMemFree((void **)&dtmp);


	delete [] htmp;
	delete [] hboxmap;
	delete [] hdisp;
	delete [] hsumRaw;
	delete [] hsumCorr;
	delete [] hFSCRaw0;
	delete [] hFSCRaw1;
	delete [] hFSCCorr0;
	delete [] hFSCCorr1;

	ResetGPU();
	pThis->Done();

	sprintf(str,"Done.\n");
	pThis->TextOutput(str);

	return (void *)0;
}
