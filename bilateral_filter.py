import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

# The CUDA kernel code, updated for 1D array usage.
kernel_code = r'''
extern "C" __global__ void BilateralFiltering(
    const unsigned short* depthImage,
    unsigned short* depthImageFiltered,
    bool* vertexValidityMask,
    const unsigned int width,
    const unsigned int height,
    const int kernelSize,
    const double spatialFactor,
    const double rangeFactor,
    const unsigned int minDepth,
    const unsigned int maxDepth)
{
    // Compute global thread coordinates.
    const unsigned int globalX = threadIdx.x + blockIdx.x * blockDim.x;
    const unsigned int globalY = threadIdx.y + blockIdx.y * blockDim.y;

    // Out-of-bound check.
    if (globalX >= width || globalY >= height)
        return;

    const int idx = globalY * width + globalX;
    const unsigned short centerValue = depthImage[idx];

    // Mark invalid if center depth is 0.
    if (centerValue == 0)
    {
        vertexValidityMask[idx] = false;
        return;
    }

    double normSum = 0.0;
    double filteredValue = 0.0;

    // Loop over a window centered on the current pixel.
    for (int yOff = -kernelSize; yOff <= kernelSize; yOff++)
    {
        for (int xOff = -kernelSize; xOff <= kernelSize; xOff++)
        {
            const int neighborX = globalX + xOff;
            const int neighborY = globalY + yOff;

            // Skip out-of-bounds neighbors.
            if (neighborX < 0 || neighborX >= width || neighborY < 0 || neighborY >= height)
                continue;

            const int n_idx = neighborY * width + neighborX;
            const unsigned short neighborValue = depthImage[n_idx];

            // Skip invalid depth values.
            if (neighborValue == 0)
                continue;

            double t_spatial_squared = double(xOff * xOff + yOff * yOff);
            double t_range_squared = double((centerValue - neighborValue) * (centerValue - neighborValue));
            double spatialWeight = exp(-0.5 * t_spatial_squared * spatialFactor);
            double rangeWeight = exp(-0.5 * t_range_squared * rangeFactor);
            double weight = spatialWeight * rangeWeight;

            normSum += weight;
            filteredValue += weight * neighborValue;
        }
    }

    if (normSum > 0.0)
        filteredValue /= normSum;
    else
        filteredValue = centerValue;

    // Clamp the filtered value between minDepth and maxDepth.
    filteredValue = fmax((double)minDepth, fmin((double)filteredValue, (double)maxDepth));

    // Ensure the filtered value is within the uint16 range.
    depthImageFiltered[idx] = (unsigned short)filteredValue;
    vertexValidityMask[idx] = true;
}

'''

# Compile the kernel.
mod = SourceModule(kernel_code)
bilateral_filtering = mod.get_function("BilateralFiltering")


def launch_bilateral_filtering_kernel(
        depth_image: np.ndarray,
        sigmaSpatial,
        sigmaRange,
        minDepth,
        maxDepth,
        width,
        height):
    """
    Applies bilateral filtering to the input depth_image using the given parameters.

    Parameters:
      depth_image    : 2D numpy array of type np.uint16.
      sigmaSpatial   : Spatial sigma (unsigned integer value).
      sigmaRange     : Range sigma (unsigned integer value).
      minDepth       : Minimum allowed depth value.
      maxDepth       : Maximum allowed depth value.

    Returns:
      A tuple (depth_image_filtered, vertex_validity_mask), where:
        - depth_image_filtered is a filtered copy of the depth image.
        - vertex_validity_mask is a boolean array (True for valid pixels).
    """
    # Flatten the depth_image (2D to 1D)
    depth_image_flat = depth_image.flatten()

    # Allocate output arrays on the host.
    depth_image_filtered = np.empty_like(depth_image_flat)
    vertex_validity_mask = np.ones_like(depth_image_flat, dtype=np.bool)

    # Allocate device memory.
    d_depth_image = cuda.mem_alloc(depth_image_flat.nbytes)
    d_depth_image_filtered = cuda.mem_alloc(depth_image_filtered.nbytes)
    d_vertex_validity_mask = cuda.mem_alloc(vertex_validity_mask.nbytes)

    # Copy input and initial output arrays to the device.
    cuda.memcpy_htod(d_depth_image, depth_image_flat)
    cuda.memcpy_htod(d_depth_image_filtered, depth_image_filtered)
    cuda.memcpy_htod(d_vertex_validity_mask, vertex_validity_mask)

    # Set the kernel parameters.
    kernelSize = 5  # Adjust kernel size based on your needs.
    spatialFactor = 1.0 / (sigmaSpatial * sigmaSpatial)
    rangeFactor = 1.0 / (sigmaRange * sigmaRange)

    # Define block and grid dimensions.
    threads_per_block = (16, 16, 1)
    blocks_per_grid_x = (width + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (height + threads_per_block[1] - 1) // threads_per_block[1]
    grid = (blocks_per_grid_x, blocks_per_grid_y, 1)

    # Launch the bilateral filtering kernel.
    bilateral_filtering(
        d_depth_image,
        d_depth_image_filtered,
        d_vertex_validity_mask,
        np.uint32(width),
        np.uint32(height),
        np.int32(kernelSize),
        np.float64(spatialFactor),
        np.float64(rangeFactor),
        np.uint32(minDepth),
        np.uint32(maxDepth),
        block=threads_per_block,
        grid=grid
    )

    # Synchronize to ensure kernel completion.
    cuda.Context.synchronize()

    # Copy results back from the device.
    cuda.memcpy_dtoh(depth_image_filtered, d_depth_image_filtered)
    cuda.memcpy_dtoh(vertex_validity_mask, d_vertex_validity_mask)

    # Free device memory.
    d_depth_image.free()
    d_depth_image_filtered.free()
    d_vertex_validity_mask.free()

    # Reshape the result back to 2D
    depth_image_filtered_reshaped = depth_image_filtered.reshape((height, width))
    vertex_validity_mask_reshaped = vertex_validity_mask.reshape((height, width))

    return depth_image_filtered_reshaped, vertex_validity_mask_reshaped
