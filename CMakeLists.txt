project(MotionCorr)

cmake_minimum_required(VERSION 2.6)
find_package(CUDA QUIET REQUIRED)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3)

# taken from EMAN2 CMakeLists.txt
MACRO(CHECK_REQUIRED_LIB2 upper lower header lower2 header2)
    FIND_PATH(${upper}_INCLUDE_PATH ${header} $ENV{${upper}DIR}/include ${INCLUDE_DIRS})
    IF(${upper}_INCLUDE_PATH)
    ELSE (${upper}_INCLUDE_PATH)
		FIND_PATH(${upper}_INCLUDE_PATH ${header2} $ENV{${upper}DIR}/include ${INCLUDE_DIRS})
    ENDIF(${upper}_INCLUDE_PATH)
    
    IF(${upper}_INCLUDE_PATH)
		FIND_LIBRARY(${upper}_LIBRARY NAMES ${lower} ${lower2} PATHS $ENV{${upper}DIR}/lib ${LIB_DIRS})		
		IF(${upper}_LIBRARY)
		ELSE(${upper}_LIBRARY)
			MESSAGE(SEND_ERROR "ERROR: ${upper} not found. please install ${upper} first!")
		ENDIF(${upper}_LIBRARY)
    ELSE(${upper}_INCLUDE_PATH)
		MESSAGE(SEND_ERROR "ERROR: ${upper} not found. please install ${upper} first!")
    ENDIF(${upper}_INCLUDE_PATH)
ENDMACRO(CHECK_REQUIRED_LIB2)

MACRO(CHECK_REQUIRED_LIB upper lower header)
    CHECK_REQUIRED_LIB2(${upper} ${lower} ${header} "" "")
ENDMACRO(CHECK_REQUIRED_LIB)

CHECK_REQUIRED_LIB(FFTW fftw3f fftw3.h)
SET(FFTW_LIBRARIES ${FFTW_LIBRARY})
SET(FFTW_INCLUDE_PATH ${FFTW_INCLUDE_PATH})

include_directories(
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/SP++3/include
    ${FFTW_INCLUDE_PATH}
)

cuda_add_executable(
    dosefgpu_driftcorr
    src/dosefgpu_driftcorr.cpp
    src/cufunc.cu
    src/DFAlign.cpp
    src/dim.cpp
    src/func.cpp
    src/mrc.cpp
    src/safefft.cpp
)

cuda_add_executable(
    gpuinfo
    src/gpuinfo.cu
)

target_link_libraries(
    dosefgpu_driftcorr
    m
    pthread
	${FFTW_LIBRARIES}
    ${CUDA_CUDA_LIBRARY}
    ${CUDA_CUFFT_LIBRARIES}
)

OPTION(ENABLE_EMAN2 "enable EMAN2 support" OFF)
IF(ENABLE_EMAN2)
    CHECK_REQUIRED_LIB(EMAN2 EM2 emdata.h)
    ADD_DEFINITIONS(-DEMAN2)
    SET(EMAN_LIBRARIES ${EMAN2_LIBRARY})
    INCLUDE_DIRECTORIES(${EMAN2_INCLUDE_PATH})
	target_link_libraries(
		dosefgpu_driftcorr
		${EMAN_LIBRARIES}	
)
ENDIF(ENABLE_EMAN2)


