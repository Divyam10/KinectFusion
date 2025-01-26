cmake_minimum_required(VERSION 3.8)

project("Measurement" LANGUAGES CXX CUDA)

set(Libs ${CMAKE_SOURCE_DIR}/Libs)

set(CMAKE_PREFIX_PATH "${Libs}/pybind11")
set(PYBIND11_FINDPYTHON ON)

find_package(CUDAToolkit REQUIRED)
find_package(Python COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG REQUIRED)

set(CMAKE_CUDA_ARCHITECTURES 60 70 80)

# This enables debugging for CUDA code (and is bad for performance)
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G -g")

# Include directories
include_directories(
    "${Libs}/OpenNI2/Include",
    ${CUDAToolkit_INCLUDE_DIRS},
    "${Libs}/eigen-3.4.0",
    "${Libs}/pybind11-master/include"
)

link_directories("${Libs}/OpenNI2/Lib" ${CUDAToolkit_LIBRARY_DIR})

# Create the Pybind11 module
pybind11_add_module(MeasurementModule MODULE Bindings.cpp)

target_link_libraries(MeasurementModule PRIVATE OpenNI2 CUDA::cudart)

target_sources(MeasurementModule PRIVATE 
    
    "Measurement.h"
    "PrimeSenseListener.h"
    "PrimeSenseVideoStreamListener.h"
    "Structs.h"
    "KinectFusion.h"   
    "kernel.cu"
    "ProcessedFrameQueue.h"
    "Constants.h"  
)

# Install to different folders
#install(TARGETS MeasurementModule
#    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib  # Shared libraries (.so, .dylib)
#    ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib  # Static libraries (.a, .lib)
#    RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin  # Runtime DLLs, executables
#)
