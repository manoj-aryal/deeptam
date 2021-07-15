cmake_minimum_required( VERSION 3.5.1 )

project(lmbspecialops)
enable_testing()

if( NOT CMAKE_BUILD_TYPE )
        set( CMAKE_BUILD_TYPE "Release" CACHE STRING "Build configuration 'Release' or 'Debug'." FORCE )
endif()

option( BUILD_WITH_CUDA "If enabled builds cuda implementation of ops"  ON )
include_directories("/home/docker/miniconda2/envs/deeptam/lib/python3.5/site-packages/tensorflow/include/")
include_directories("/home/docker/miniconda2/envs/deeptam/lib/python3.5/site-packages/external/nsync/public/")

# enable all warnings
set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -D_GLIBCXX_USE_CXX11_ABI=0" )
set( CUDA_SDK_ROOT_DIR "/usr/local/cuda/samples" )
set( GENERATE_KEPLER_SM35_CODE "ON")
set( GENERATE_KEPLER_SM37_CODE "ON")
set( GENERATE_MAXWELL_SM52_CODE "ON")
set( GENERATE_PASCAL_SM60_CODE "ON")
set( GENERATE_PASCAL_SM61_CODE "ON")
set( GENERATE_PTX61_CODE "ON")

add_subdirectory( src lib ) 
add_subdirectory( doc ) 
add_subdirectory( test test ) 

