This Git repositiory of MotioCorr is based on the MotionCorr 2.0 release with the following extensions:
        1. Read any image format (DM4, mrcs, and essentially all formats used in cryo-EM fields by using EMAN2 IO functions)
        2. Automatically convert to native endian of current computer regardless of starting endianness (i.e. big-endian used by DM4 images to little-endian used by Intel CPUs)
        3. Compile on all major computer systems (Linux, Mac, and Windows) using the cross-platform CMake build system

**********************************************************************
The contents below are original contents included in MotionCorr 2.0
**********************************************************************

Motion Correction for Dose-Fractionation Stack
Version 1.0, April 11, 2013


Requirement:
    GPU hardware version: >2.0
    GPU memory: >1.5Gb
    CUDA 5.0 or higher
    g++ 4.6 or lower


Compile:
    Run "make" in src folder
 
Features:
GPU enabled. The computations are accelerated by GPU. Processing a stack with
20~30frames only takes ~20s, and usually less than 1 minute for more frames,
hence is suit for on-the-fly processing.

    
Run program:
    1) Run gpuinfo in bin folder to get a list of availabe GPU. If device 0 isn't the GPU you want to use, you need to specify its ID. See the following example.
    2) Run dosefgpu_driftcorr without fowllowing any parameters to get a help.
    3) To process your K2 image, see the following example:
    
       Example 1:   dosefgpu_driftcorr yourstack.mrc
       Example 2:   dosefgpu_driftcorr yourstack.mrc -bin 2 -bft 150 -scc 1
       
    4) You can run dosef_logviewer in the folder you run the program to check the result. 
       
    The program has a set of default output filename, so you don't need to set filename. But you need put all stack file into the same folder you run the program. Or you can specify the output filename in paramereters.
    The binning uses Fourier cropping, so no worry to the binning effects.
    
Note:
    1) A bfactor 150 or 200pix^2 is good for most cryoEM image with 2x binned super-resolution image. For unbined image, a larger bfactor is needed.
    2) Suggest to use more than 20 frames.   
    3) -fod defines the minmum distance of frame comparision, which counts from the middle frame. Suggest to set -fod to TotalNumFrame/4
    
The Motion Correction program was written by Xueming Li at Yifan Cheng
Laboratory at UCSF. For more details, source code and future updates, please visit the
lab website at http://cryoem.ucsf.edu , or contact the author directly.

The Motion Correction program are licensed under the terms of the GNU Public
License version 3 (GPLv3).


***************************************
Update:
07.01.2013  Fix a bug of "align to middle"

08.26.2013  Fix bugs and add new functions:		
				1) support rectangular image, replace -crd with -cdx and -cdy
				2) fix bug of "align to middle", and now can align to any specified frame 
				3) output un-corrected stack
				4) remove -sub option
				5) adjust output function to make the program more stable
				6) the min,max,mean value of output MRC stacks are set to the values of last frame
				
08.31.2013  Add new functions:
            1) Gain and dark normalization
            2) Add non-standard MRC format: mode 5(unsigned 8-bit integer) and mode 6(unsigned 16-bit integer)

***************************************
