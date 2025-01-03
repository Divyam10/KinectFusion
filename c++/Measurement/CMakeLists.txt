cmake_minimum_required(VERSION 3.8)

project("Measurement")

# Include directories
include_directories("C:/Program Files/OpenNI2/Include")
link_directories("C:/Program Files/OpenNI2/Lib")

# Create a testing executable
add_executable(measurement_executable "Measurement.cpp" "Measurement.h" "PrimeSenseListener.h" "PrimeSenseVideoStreamListener.h" "FrameTuple.h" "KinectFusion.h"  "ImageData.h")

# Link libraries
target_link_libraries(measurement_executable OpenNI2)

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
