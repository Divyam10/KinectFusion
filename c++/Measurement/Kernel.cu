#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <memory>
#include <iostream>

//TODO Optimizations: shared memory and precomputing of euclidean distances
__global__ void BilateralFiltering(
    const uint16_t* depthImage,
    uint16_t* depthImageFiltered,
    const unsigned int width,
    const unsigned int height,
    const int kernelSize,
    const double spatialFactor,
    const double rangeFactor,
    const unsigned int minDepth,
    const unsigned int maxDepth)
{
    const unsigned int globalX = threadIdx.x + blockIdx.x * blockDim.x;
    const unsigned int globalY = threadIdx.y + blockIdx.y * blockDim.y;


    // Return if pixel is out of bounds
    if (globalX >= width || globalY >= height)
    {
        return;
    }

    const uint16_t centerValue = depthImage[globalY * width + globalX];
    double normSum = 0;
    double filteredValue = 0;

    // Apply bilateral filtering over a window centered around the current pixel
    for (int yOff = -kernelSize; yOff <= kernelSize; yOff++)
    {
        for (int xOff = -kernelSize; xOff <= kernelSize; xOff++)
        {
            const int neighborX = globalX + xOff;
            const int neighborY = globalY + yOff;

            if (neighborX < 0 || neighborX >= width || neighborY < 0 || neighborY >= height)
            {
                continue;
            }

            const int neighborGloablIdx = neighborY * width + neighborX;
            const uint16_t neighborValue = depthImage[neighborGloablIdx];

            const double t_spatial_squared = double((xOff * xOff) + (yOff * yOff));
            const double t_range_squared = double((centerValue - neighborValue) * (centerValue - neighborValue));

            const double spatialWeight = exp(-1.0 * t_spatial_squared * spatialFactor);
            const double rangeWeight = exp(-1.0 * t_range_squared * rangeFactor);


            const double weight = spatialWeight * rangeWeight;

            normSum += weight;
            filteredValue += (weight * neighborValue);
        }
    }

    if (normSum > 0)
        filteredValue /= normSum;
    else
        filteredValue = double(centerValue);

    filteredValue = max(minDepth, min(unsigned int(filteredValue), maxDepth));

    depthImageFiltered[globalY * width + globalX] = uint16_t(filteredValue);
}



// C++ CUDA wrapper
extern "C" void LaunchBilateralFilteringKernel(
    const uint16_t* depthImage,
    uint16_t* depthImageFiltered,
    const unsigned int pixelCount,
    const unsigned int width,
    const unsigned int height,
    const unsigned int minDepth,
    const unsigned int maxDepth)
{

    const unsigned int sigmaSpatial = 5;
    const float sigmaRange = 50.f;
    const unsigned int kernelSize = 21;

    const double spatialFactor = 1.0 / double(sigmaSpatial * sigmaSpatial);
    const double rangeFactor = 1.0 / double(sigmaRange * sigmaRange);

    const dim3 threadsPerBlock(16, 16);
    const dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    uint16_t* d_depthImage;
    uint16_t* d_depthImageFiltered;

    auto err = cudaMalloc(&d_depthImage, pixelCount * sizeof(uint16_t));
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc failed for d_depthImage: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    err = cudaMalloc(&d_depthImageFiltered, pixelCount * sizeof(uint16_t));
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc failed for d_depthImageFiltered: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    err = cudaMemcpy(d_depthImage, depthImage, pixelCount * sizeof(uint16_t), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "CUDA memcpy failed for d_depthImage: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    err = cudaMemcpy(d_depthImageFiltered, depthImageFiltered, pixelCount * sizeof(uint16_t), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "CUDA memcpy failed for d_depthImageFiltered: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    // Launch the bilateral filtering kernel
    BilateralFiltering <<< numBlocks, threadsPerBlock >>> (d_depthImage, d_depthImageFiltered, width, height, kernelSize, spatialFactor, rangeFactor, minDepth, maxDepth);

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "CUDA synchronization failed: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    err = cudaMemcpy(depthImageFiltered, d_depthImageFiltered, pixelCount * sizeof(uint16_t), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "CUDA memcpy back failed for d_depthImageFiltered: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    err = cudaFree(d_depthImage);
    if (err != cudaSuccess) {
        std::cerr << "CUDA free failed for d_depthImage: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    err = cudaFree(d_depthImageFiltered);
    if (err != cudaSuccess) {
        std::cerr << "CUDA free failed for d_depthImageFiltered: " << cudaGetErrorString(err) << std::endl;
        return;
    }
}
