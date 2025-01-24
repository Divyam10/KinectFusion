#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <memory>
#include <iostream>

//TODO Optimizations: shared memory and precomputing of euclidean distances
__global__ void BilateralFiltering(
    const uint16_t* depthImage,
    uint16_t* depthImageFiltered,
    bool* vertexValidityMask,
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

    //Invalid depth value
    if (centerValue == 0)
    {
        vertexValidityMask[globalY * width + globalX] = false;
        return;
    }

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

            const double spatialWeight = exp(-0.5f * t_spatial_squared * spatialFactor);
            const double rangeWeight = exp(-0.5f * t_range_squared * rangeFactor);


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
    bool* vertexValidityMask,
    const unsigned int pixelCount,
    const unsigned int width,
    const unsigned int height,
    const unsigned int minDepth,
    const unsigned int maxDepth,
    const unsigned int sigmaSpatial,
    const unsigned int sigmaRange)
{
    const unsigned int kernelSize = 21;

    const double spatialFactor = 1.0 / double(sigmaSpatial * sigmaSpatial);
    const double rangeFactor = 1.0 / double(sigmaRange * sigmaRange);

    const dim3 threadsPerBlock(16, 16);
    const dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    uint16_t* d_depthImage;
    uint16_t* d_depthImageFiltered;
    bool* d_vertexValidityMask;

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

    err = cudaMalloc(&d_vertexValidityMask, pixelCount * sizeof(bool));
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc failed for d_vertexValidityMask: " << cudaGetErrorString(err) << std::endl;
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

    err = cudaMemcpy(d_vertexValidityMask, vertexValidityMask, pixelCount * sizeof(bool), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "CUDA memcpy failed for d_vertexValidityMask: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    // Launch the bilateral filtering kernel
    BilateralFiltering <<< numBlocks, threadsPerBlock >>> (d_depthImage, d_depthImageFiltered, d_vertexValidityMask, width, height, kernelSize, spatialFactor, rangeFactor, minDepth, maxDepth);

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

    err = cudaMemcpy(vertexValidityMask, d_vertexValidityMask, pixelCount * sizeof(bool), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "CUDA memcpy back failed for d_vertexValidityMask: " << cudaGetErrorString(err) << std::endl;
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

    err = cudaFree(d_vertexValidityMask);
    if (err != cudaSuccess) {
        std::cerr << "CUDA free failed for d_vertexValidityMask: " << cudaGetErrorString(err) << std::endl;
        return;
    }
}

__global__ void BackProjection(
    const uint16_t* depthImageFiltered,
    float* K_inv,
    uint16_t* backProjectedDepthImage,
    const unsigned int width,
    const unsigned int height,
    const unsigned int stride)
{
    const unsigned int pixelU = threadIdx.x + blockIdx.x * blockDim.x;
    const unsigned int pixelV = threadIdx.y + blockIdx.y * blockDim.y;

    // Return if pixel is out of bounds
    if (pixelU >= width || pixelV >= height)
    {
        return;
    }

    const float pixelDepthValue = float(depthImageFiltered[pixelV * width + pixelU]);
    const float newX = pixelDepthValue * (K_inv[0] * float(pixelU) + K_inv[2]);
    const float newY = pixelDepthValue * (K_inv[4] * float(pixelV) + K_inv[5]);
    const float newZ = pixelDepthValue;

    backProjectedDepthImage[pixelV * width + stride * pixelU] = newX;
    backProjectedDepthImage[pixelV * width + stride * pixelU + 1] = newY;
    backProjectedDepthImage[pixelV * width + stride * pixelU + 2] = newZ;
}




extern "C" void LaunchBackProjectionKernel(
    float* K_inv,
    const uint16_t* depthImageFiltered,
    uint16_t* backProjectedDepthImage,
    const unsigned int pixelCount,
    const unsigned int width,
    const unsigned int height,
    const unsigned int stride)
{
    const dim3 threadsPerBlock(16, 16);
    const dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    uint16_t* d_depthImageFiltered;
    uint16_t* d_backProjectedDepthImage;
    float* d_K_inv;


    auto err = cudaMalloc(&d_depthImageFiltered, pixelCount * sizeof(uint16_t));
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc failed for d_depthImageFiltered: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    err = cudaMalloc(&d_backProjectedDepthImage, stride * pixelCount * sizeof(uint16_t));
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc failed for d_backProjectedDepthImage: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    err = cudaMalloc(&d_K_inv, 9 * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc failed for d_K_inv: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    err = cudaMemcpy(d_depthImageFiltered, depthImageFiltered, pixelCount * sizeof(uint16_t), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "CUDA memcpy failed for d_depthImageFiltered: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    err = cudaMemcpy(d_backProjectedDepthImage, backProjectedDepthImage, stride * pixelCount * sizeof(uint16_t), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "CUDA memcpy failed for d_backProjectedDepthImage: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    err = cudaMemcpy(d_K_inv, K_inv, 9 * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "CUDA memcpy failed for d_K_inv: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    // Launch the backprojection kernel
    BackProjection <<< numBlocks, threadsPerBlock >> > (d_depthImageFiltered, d_K_inv, d_backProjectedDepthImage, width, height, stride);

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "CUDA synchronization failed: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    err = cudaMemcpy(backProjectedDepthImage, d_backProjectedDepthImage, stride * pixelCount * sizeof(uint16_t), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "CUDA memcpy back failed for d_backProjectedDepthImage: " << cudaGetErrorString(err) << std::endl;
        return;
    }
    
    err = cudaFree(d_backProjectedDepthImage);
    if (err != cudaSuccess) {
        std::cerr << "CUDA free failed for d_backProjectedDepthImage: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    err = cudaFree(d_depthImageFiltered);
    if (err != cudaSuccess) {
        std::cerr << "CUDA free failed for d_depthImageFiltered: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    err = cudaFree(d_K_inv);
    if (err != cudaSuccess) {
        std::cerr << "CUDA free failed for d_K_inv: " << cudaGetErrorString(err) << std::endl;
        return;
    }

}


__global__ void BlockAveragingAndSubsampling(
    const uint16_t* depthImage,
    uint16_t* depthImageAveraged,
    const unsigned int width,
    const unsigned int height,
    const unsigned int sigmaRange,
    const unsigned int blockSize)
{
    const unsigned int globalX = threadIdx.x + blockIdx.x * blockDim.x;
    const unsigned int globalY = threadIdx.y + blockIdx.y * blockDim.y;
    const int similarDepthRange = 3.f * float(sigmaRange);


    // Return if pixel is out of bounds
    if (globalX >= width || globalY >= height) 
    {
        return;
    }

    //TODO instead use less threads
    if (globalX % 2 != 0 || globalY % 2 != 0)
    {
        return;
    }
    //Assume they use top left pixel as "central pixel". Could also be using precalc. mean?
    const uint16_t topLeftValue = depthImage[globalY * width + globalX];
    const uint16_t topRightValue = depthImage[globalY * width + globalX + 1];
    const uint16_t bottomLeftValue = depthImage[(globalY + 1) * width + globalX];
    const uint16_t bottomRightValue = depthImage[(globalY + 1) * width + globalX + 1];

    unsigned int sum = 0;
    float norm = 0.f;

    sum += topLeftValue;
    norm++;

    if (abs(topRightValue - topLeftValue) <= similarDepthRange) {
        sum += topRightValue;
        norm++;
    }
    if (abs(bottomLeftValue - topLeftValue) <= similarDepthRange) {
        sum += bottomLeftValue;
        norm++;
    }
    if (abs(bottomRightValue - topLeftValue) <= similarDepthRange) {
        sum += bottomRightValue;
        norm++;
    }

    const uint16_t average = (norm > 0) ? uint16_t(float(topLeftValue + topRightValue + bottomLeftValue + bottomRightValue) / norm) : 0;

    depthImageAveraged[(globalY / 2) * width + (globalX / 2)] = average;
}


extern "C" void LaunchBlockAveragingAndSubsampleKernel(
    const uint16_t* depthImageL,
    uint16_t* depthImageLAveraged,
    const unsigned int pixelCountL,
    const unsigned int pixelCountLAveraged,
    const unsigned int widthL,
    const unsigned int heightL,
    const unsigned int blockSize,
    const unsigned int sigmaRange)
{
    const dim3 threadsPerBlock(16, 16);
    const dim3 numBlocks((widthL + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (heightL + threadsPerBlock.y - 1) / threadsPerBlock.y);

    uint16_t* d_depthImageL;
    uint16_t* d_depthImageLAveraged;

    auto err = cudaMalloc(&d_depthImageL, pixelCountL * sizeof(uint16_t));
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc failed for d_depthImageL: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    err = cudaMalloc(&d_depthImageLAveraged, pixelCountLAveraged * sizeof(uint16_t));
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc failed for d_depthImageLAveraged: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    err = cudaMemcpy(d_depthImageL, depthImageL, pixelCountL * sizeof(uint16_t), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "CUDA memcpy failed for d_depthImageL: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    err = cudaMemcpy(d_depthImageLAveraged, depthImageLAveraged, pixelCountLAveraged * sizeof(uint16_t), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "CUDA memcpy failed for d_depthImageLAveraged: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    // Launch the BlockAveragingAndSubsampling kernel
    BlockAveragingAndSubsampling <<< numBlocks, threadsPerBlock >>> (d_depthImageL, d_depthImageLAveraged, widthL, heightL, sigmaRange, blockSize);

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "CUDA synchronization failed: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    err = cudaMemcpy(depthImageLAveraged, d_depthImageLAveraged, pixelCountLAveraged * sizeof(uint16_t), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "CUDA memcpy failed for d_depthImageLAveraged: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    err = cudaFree(d_depthImageLAveraged);
    if (err != cudaSuccess) {
        std::cerr << "CUDA free failed for d_depthImageLAveraged: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    err = cudaFree(d_depthImageL);
    if (err != cudaSuccess) {
        std::cerr << "CUDA free failed for d_depthImageL: " << cudaGetErrorString(err) << std::endl;
        return;
    }
}

__global__ void NormalMap(
    const uint16_t* vertexMap,
    uint16_t* normalMap,
    const unsigned int width,
    const unsigned int height)
{
    const unsigned int globalX = threadIdx.x + blockIdx.x * blockDim.x;
    const unsigned int globalY = threadIdx.y + blockIdx.y * blockDim.y;

    // Return if pixel is out of bounds or if pixel u+1 or v+1 out of bounds
    //TODO globalX%3 -> instead less threads!
    if (globalX >= width - 1 || globalY >= height - 1 || globalX % 3 != 0)
    {
        return;
    }

    uint16_t centerValueX = vertexMap[globalY * width + globalX];
    uint16_t centerValueY = vertexMap[globalY * width + globalX + 1];
    uint16_t centerValueZ = vertexMap[globalY * width + globalX + 2];

    uint16_t rightValueX = vertexMap[globalY * width + globalX + 3];
    uint16_t rightValueY= vertexMap[globalY * width + globalX + 4];
    uint16_t rightValueZ = vertexMap[globalY * width + globalX + 5];

    //TODO check globalY (not just here)
    uint16_t belowValueX = vertexMap[(globalY + 1) * width + globalX];
    uint16_t belowValueY = vertexMap[(globalY + 1) * width + globalX + 1];
    uint16_t belowValueZ = vertexMap[(globalY + 1) * width + globalX + 2];

    uint16_t xDiffX = rightValueX - centerValueX;
    uint16_t xDiffY = rightValueY - centerValueY;
    uint16_t xDiffZ = rightValueZ - centerValueZ;

    uint16_t yDiffX = belowValueX - centerValueX;
    uint16_t yDiffY = belowValueY - centerValueY;
    uint16_t yDiffZ = belowValueZ - centerValueZ;

    normalMap[globalY * width + globalX] = xDiffY * yDiffZ - xDiffZ * yDiffY;
    normalMap[globalY * width + globalX + 1] = xDiffZ * yDiffX - xDiffX * yDiffZ;
    normalMap[globalY * width + globalX + 2] = xDiffX * yDiffY - xDiffY * yDiffX;
}

extern "C" void LaunchNormalMapKernel(
    const uint16_t* vertexMap,
    uint16_t* normalMap,
    const unsigned int pixelCount,
    const unsigned int width,
    const unsigned int height,
    const unsigned int stride)
{
    const dim3 threadsPerBlock(16, 16);
    const dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    uint16_t* d_vertexMap;
    uint16_t* d_normalMap;

    auto err = cudaMalloc(&d_vertexMap, pixelCount * stride *  sizeof(uint16_t));
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc failed for d_vertexMap: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    err = cudaMalloc(&d_normalMap, pixelCount * stride * sizeof(uint16_t));
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc failed for d_normalMap: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    err = cudaMemcpy(d_vertexMap, vertexMap, pixelCount * stride * sizeof(uint16_t), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "CUDA memcpy failed for d_vertexMap: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    err = cudaMemcpy(d_normalMap, normalMap, pixelCount * stride * sizeof(uint16_t), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "CUDA memcpy failed for d_normalMap: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    // Launch the BlockAveragingAndSubsampling kernel
    NormalMap << < numBlocks, threadsPerBlock >> > (d_vertexMap, d_normalMap, width, height);

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "CUDA synchronization failed: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    err = cudaMemcpy(normalMap, d_normalMap, pixelCount * stride * sizeof(uint16_t), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "CUDA memcpy failed for d_normalMap: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    err = cudaFree(d_normalMap);
    if (err != cudaSuccess) {
        std::cerr << "CUDA free failed for d_normalMap: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    err = cudaFree(d_vertexMap);
    if (err != cudaSuccess) {
        std::cerr << "CUDA free failed for d_vertexMap: " << cudaGetErrorString(err) << std::endl;
        return;
    }
}
