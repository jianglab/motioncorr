project(MotionCorr)

cmake_minimum_required(VERSION 2.6)
find_package(CUDA QUIET REQUIRED)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3)

include_directories( 
    src
    src/SP++3/include
)

link_directories(
    src/lib
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
    fftw3f
    m
    pthread
    cuda
    cufft
)