cmake_minimum_required (VERSION 3.8 FATAL_ERROR)

project (GPHash)

find_package(CUDA 8.0 REQUIRED)

option(CMAKE_VERBOSE_MAKEFILE ON)

set(CUDA_NVCC_FLAGS -std=c++11 -O3)
set(CMAKE_CXX_STANDARD 11)

if (CUDA_VERBOSE_PTXAS)
  set(VERBOSE_PTXAS --ptxas-options=-v)
endif (CUDA_VERBOSE_PTXAS)

set(CMAKE_BUILD_TYPE "Release")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(GENCODE_SM70
  -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_70,code=compute_70)

set(GENCODE ${GENCODE} ${GENCODE_SM70})

include_directories(src)

add_subdirectory(test)