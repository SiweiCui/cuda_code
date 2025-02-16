cmake_minimum_required(VERSION 3.22)

# Set the CUDA compiler path or name
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")

project(cuda_code CXX CUDA)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)

find_package(GTest REQUIRED)
find_package(glog REQUIRED)

set(link_ext_lib glog::glog GTest::gtest GTest::gtest_main)

aux_source_directory(. DIR_TEST)

add_executable(cuda_code_test ${DIR_TEST})

target_link_libraries(cuda_code_test PUBLIC ${link_ext_lib})

target_include_directories(cuda_code_test PUBLIC ./include)
target_include_directories(cuda_code_test PUBLIC ${glog_INCLUDE_DIR})
target_include_directories(cuda_code_test PUBLIC ${GTest_INCLUDE_DIR})

