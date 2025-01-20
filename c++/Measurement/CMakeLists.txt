cmake_minimum_required(VERSION 3.8)

project("Measurement" LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CUDA_ARCHITECTURES 60 70 80)

# Somehow this enables debugging for CUDA code (and is bad for performance)
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G -g")

# Include directories
include_directories(
    "C:/Program Files/OpenNI2/Include",
    ${CUDAToolkit_INCLUDE_DIRS},
    "C:/Users/steph/Documents/Projekte/KinectFusion/c++/Libs/eigen-3.4.0",
    "C:/Users/steph/Documents/Projekte/KinectFusion/c++/Libs/pybind11-master/include")
link_directories("C:/Program Files/OpenNI2/Lib" ${CUDAToolkit_LIBRARY_DIR})

# Create a testing executable
add_executable(measurement_executable "Measurement.cpp" "Measurement.h" "PrimeSenseListener.h" "PrimeSenseVideoStreamListener.h" "FrameTuple.h" "KinectFusion.h" "ImageData.h" "kernel.cu" "QueueWrapper.h" "Constants.h")

# Link libraries
target_link_libraries(measurement_executable OpenNI2 CUDA::cudart)

# Gathers all files in Redist
set(OPENNI2_SOURCE_DIR "C:/Program Files/OpenNI2/Redist")
file(GLOB_RECURSE OPENNI2_FILES RELATIVE ${OPENNI2_SOURCE_DIR} "${OPENNI2_SOURCE_DIR}/*")

# Copy files to the destination directory
foreach(FILE ${OPENNI2_FILES})
    # Absolute source path
    set(SOURCE_PATH "${OPENNI2_SOURCE_DIR}/${FILE}")
    
    # Copy the files
    add_custom_command(TARGET measurement_executable POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${SOURCE_PATH}" "${CMAKE_BINARY_DIR}/$<CONFIG>/${FILE}"
    )
endforeach()
