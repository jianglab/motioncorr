#include <cuda.h>
#include <stdio.h>

int main()
{
	printf("\nSimple GPU info query.\n\n");
	int ngpu;
	cudaGetDeviceCount(&ngpu);
	if(ngpu <= 0)
	{
		printf("Didn't found usable nVidia GPGPU.\n");
		return 0;
	}
	else
	{
		printf("You have %d nVidia GPGPU.\n\n",ngpu);
	}

	printf("DeviceID Name\t\t\t Version Memory(Mb)\n");
	cudaDeviceProp prop;
	int i;
	for(i=0;i<ngpu;i++)
	{
		if(cudaGetDeviceProperties(&prop, i) == cudaSuccess) 
		{
			
			if(prop.kernelExecTimeoutEnabled)
			{
				printf("#%d\t %s\t %1.1f\t %d\t Has Monitor\n",
					i,prop.name,prop.major+prop.minor/10.0,int(prop.totalGlobalMem/1024.0/1024.0));
			}
			else printf("#%d\t %s\t %1.1f\t %d\n",
						i,prop.name,prop.major+prop.minor/10.0,int(prop.totalGlobalMem/1024.0/1024.0));
		}
		else printf("Failed to get GPU info of Device #%d\n",i);
	}
	
	printf("\nWrote by Xueming Li @ Yifan Cheng Lab, UCSF\n");

	return 0;
}
