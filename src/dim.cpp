#include "dim.h"
#include <stdio.h>


DIM::DIM()
{
	x=0;
	y=0;
}

DIM::~DIM()
{

}


DIM::DIM(int w, int h)
{
	x=w;
	y=h;
}

DIM::DIM(const DIM &d)
{
	x=d.x;
	y=d.y;
}

int DIM::width()
{
	return x;
}


int DIM::height()
{
	return y;
}

bool DIM::IsSquare()
{
	return x==y;
}

DIM DIM::MinSquare()
{
	int m = x<=y?x:y;
	return DIM(m,m); 
}


DIM DIM::MinSquareOffset()
{
	DIM m=MinSquare();
	return DIM(x-m.x,y-m.y);
}

void DIM::print()
{
	printf("  DIM(x,y):  %5d %5d \n",x,y);
}

void DIM::print(char *name)
{
	printf("  DIM(x,y):  %5d %5d  (Name: %s)\n",x,y,name);
}

DIM DIM::operator + (const DIM &d)
{
	return DIM(x+d.x,y+d.y);
}

DIM DIM::operator + (const int &d)
{
	return DIM(x+d,y+d);
}

DIM DIM::operator * (const DIM &d)
{
	return DIM(x*d.x,y*d.y);
}

DIM DIM::operator * (const int &d)
{
	return DIM(x*d,y*d);
}

DIM DIM::operator / (const DIM &d)
{
	return DIM(x/d.x,y/d.y);
}

DIM DIM::operator / (const int &d)
{
	return DIM(x/d,y/d);
}

DIM &DIM::operator = (const DIM &d)
{
	this->x=d.x;
	this->y=d.y;
	return *this;
}

DIM &DIM::operator = (const int &d)
{
	this->x=d;
	this->y=d;
	return *this;
}

bool DIM::operator == (const DIM &d)
{
	return (x==d.x && y==d.y);
}

bool DIM::operator == (const int &d)
{
	return (x==d && y==d);
}

bool DIM::operator != (const DIM &d)
{
	return (x!=d.x || y!=d.y);
}

bool DIM::operator != (const int &d)
{
	return (x!=d || y!=d);
}

bool DIM::operator > (const DIM &d)
{
	return (x>d.x || y>d.y);
}











