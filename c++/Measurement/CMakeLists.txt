cmake_minimum_required(VERSION 3.8)

project("Measurement" LANGUAGES CXX CUDA)

set(Libs "C:/Users/steph/Documents/Projekte/KinectFusion/c++/Libs")
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
    "C:/Program Files/OpenNI2/Include",  # TODO install in libs folder
    ${CUDAToolkit_INCLUDE_DIRS},
    "${Libs}/eigen-3.4.0",
    "${Libs}/pybind11-master/include"
)

link_directories("C:/Program Files/OpenNI2/Lib" ${CUDAToolkit_LIBRARY_DIR})

# Create the main executable
#add_executable(measurement_executable 
#    "Measurement.h"
#    "PrimeSenseListener.h"
#    "PrimeSenseVideoStreamListener.h" 
#    "Structs.h"
#    "KinectFusion.h"
#    "kernel.cu"
#    "ProcessedFrameQueue.h"
#    "Constants.h" 
#)
#target_link_libraries(measurement_executable PRIVATE OpenNI2 CUDA::cudart)

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

# Gathers all files in Redist
set(OPENNI2_SOURCE_DIR "C:/Program Files/OpenNI2/Redist")
file(GLOB_RECURSE OPENNI2_FILES RELATIVE ${OPENNI2_SOURCE_DIR} "${OPENNI2_SOURCE_DIR}/*")

# Copy files to the destination directory after build
foreach(FILE ${OPENNI2_FILES})
    set(SOURCE_PATH "${OPENNI2_SOURCE_DIR}/${FILE}")
    add_custom_command(TARGET MeasurementModule POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${SOURCE_PATH}" "${CMAKE_BINARY_DIR}/$<CONFIG>/${FILE}"
    )
endforeach()

# Create Module
install(TARGETS MeasurementModule DESTINATION .)

# TODO maybe transfer cuda and openni2 dlls to module location
