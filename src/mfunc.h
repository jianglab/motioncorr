#pragma once

#include <linequs2.h>

using namespace splab;
#include <complex>
#include <vector>
#include <algorithm>
using namespace std;

//int OD_SetEquation(Matrix<complex<double> > &m, vector<complex<int> > &list, int nframe);

//Version 1: don't do comparison between frames after  nleft=nframe/2
//           becasue the relative shift bewtween them may be too small
//           but the missed comparison may reduce the precision
int OD_GetNRow(int nframe, int FrameDistOffset=0)
{
	int nleft=nframe/2;
	int nright=nframe-nleft+FrameDistOffset;
	int i,n=0;
	for(i=0;i<nleft;i++)
	{
		n+=nright;
		nright--;
	}

	return n;

}
int OD_SetEquation(Matrix<complex<double> > &m, vector<complex<int> > &list, int nframe, int FrameDistOffset=0)
{
	int nrow=OD_GetNRow(nframe,FrameDistOffset);
	m=Matrix<complex<double> >(nrow,nframe-1,0.0);
	
	list.clear();
	int nleft=nframe/2;
	int i,j,k;
	for(i=0;i<nleft;i++)
		for(j=nleft+i-FrameDistOffset;j<nframe;j++)
		{
			for(k=i;k<j;k++) m[list.size()][k]=1.0;
			list.push_back(complex<int>(i,j));
		}

	return list.size();
}

//Version 2: different from Version 1, do all the possible comparison
//           so, only limit the frame distance
int OD_GetNRow_All(int nframe, int FrameDistOffset=0)
{
	int dist=nframe/2-FrameDistOffset;
	if(dist<=0) dist=1;
	if(dist>nframe/2) dist=nframe/2;
	int i,j,n=0;
	for(i=0;i<nframe;i++)
		for(j=i+dist;j<nframe;j++)
		{
			n++;
		}

	return n;
}
int OD_SetEquation_All(Matrix<complex<double> > &m, vector<complex<int> > &list, int nframe, int FrameDistOffset=0)
{
	int nrow=OD_GetNRow_All(nframe,FrameDistOffset);
	m=Matrix<complex<double> >(nrow,nframe-1,0.0);
	
	list.clear();

	int dist=nframe/2-FrameDistOffset;
	if(dist<=0) dist=1;
	if(dist>nframe/2) dist=nframe/2;

	int i,j,k;
	for(i=0;i<nframe;i++)
		for(j=i+dist;j<nframe;j++)
		{
			for(k=i;k<j;k++) m[list.size()][k]=1.0;
			list.push_back(complex<int>(i,j));
		}

	return list.size();
}

int Min_VectorIndex(Vector<double> &v)
{
	int pos=0;
	double m=v[0];
	for(int i=1;i<v.size();i++)
	{
		if(v[i]<m)
		{
			m=v[i];
			pos=i;
		}
	}

	return pos;
}
vector<int> OD_Threshold(Matrix<complex<double> > &A, Vector<complex<double> > &b, Vector<double> &ki, double kiThresh)
{

	vector<int> goodlist,bk;
	Matrix<complex<double> > tA=A;
	Vector<complex<double> > tb=b;
	Vector<double> tki=ki;
	CSVD<double> svd;
	int rank=0;
	int i,pos,ii;
	for(i=0;i<tki.size();i++)
	{
		pos=Min_VectorIndex(tki);
		goodlist.push_back(pos);
		tki[pos]=1e20;
	}
	bk=goodlist;

	for(i=goodlist.size()-1;i>=A.cols();i--)
	{
		if(ki[goodlist[i] ]<kiThresh) break;
	}
	ii=i;

	do
	{
		goodlist=bk;
		goodlist.erase(goodlist.begin()+ii+1,goodlist.end());
		sort(goodlist.begin(),goodlist.end());

		A=Matrix<complex<double> >(goodlist.size(),A.cols(),0.0);
		b=Vector<complex<double> >(goodlist.size(),0.0);
		for(i=0;i<goodlist.size();i++)
		{
			A.setRow(tA.getRow(goodlist[i]),i);
			b[i]=tb[goodlist[i] ];
		}
	
		svd.dec(A);
		rank=svd.rank();

		ii++;
	}while(rank<A.cols() && ii<bk.size());

	return goodlist;
}
