#pragma once

class DIM
{
public:
	DIM();
	DIM(int w, int h);
	DIM(const DIM &d);
	~DIM();


public:
	int x; //along row
	int y; //along column
	
public:
	int width();
	int height();
	
	bool IsSquare();
	
	//used to cut minimum square image for FFT display
	DIM MinSquare();
	DIM MinSquareOffset();
	
	void print();
	void print(char *name);
	

public:
	DIM operator + (const DIM &d);
	DIM operator + (const int &d);
	DIM operator * (const DIM &d);
	DIM operator * (const int &d);
	DIM operator / (const DIM &d);
	DIM operator / (const int &d);
	DIM & operator = (const DIM &d);
	DIM & operator = (const int &d);
	bool operator == (const DIM &d);	
	bool operator == (const int &d);
	bool operator != (const DIM &d);	
	bool operator != (const int &d);	
	bool operator > (const DIM &d);	


};
