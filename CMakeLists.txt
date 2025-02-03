cmake_minimum_required (VERSION 3.5)

# set(CMAKE_BUILD_TYPE Debug) # Do not set if profiling using NSight
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
set(CMAKE_CXX_STANDARD 11)

project (cudacnnproject)
include(FindCUDA)
# list( APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_70,code=sm_70; -std=c++11; -lineinfo")
# list( APPEND CUDA_NVCC_FLAGS "; -std=c++11; -lineinfo")
list( APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_86,code=[sm_86,compute_86]; -std=c++11; -lineinfo")

include_directories ("${PROJECT_SOURCE_DIR}/third_party/eigen")

aux_source_directory(. DIR_SRCS)

add_subdirectory(src)

add_library(cudacnnet cudacnnnet.cc cudacnnet.h)

cuda_add_executable(forward_pass forward_pass.cc)
target_link_libraries(forward_pass cudacnnet MiniDNNLib)

