import torch
import torch.nn.functional as F

def block_averaging(depth_image, block_size, sigma_range):
    unfolded = depth_image.unfold(0, block_size, block_size).unfold(1, block_size, block_size)
    # Top left pixel as reference
    top_left_values = unfolded[..., 0, 0]
    # Diff to reference
    diff = torch.abs(unfolded - top_left_values[..., None, None])
    # Edge preservance condition
    valid_mask = diff <= 3 * sigma_range
    valid_values = unfolded * valid_mask
    norm = valid_mask.sum(dim=(-2, -1))
    summed = valid_values.sum(dim=(-2, -1))

    avg_depth = summed / norm.float()

    return avg_depth.round()


'''def block_averaging(depth_image, block_size, sigma_range):
    # Ensure input is 4D (Batch=1, Channels=1, H, W)
    depth_image = depth_image.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)

    # Count valid (nonzero) values per block
    valid_counts = F.avg_pool2d((depth_image > 0).float(), kernel_size=block_size, stride=block_size, divisor_override=1)

    # Compute summed depth values per block
    summed = F.avg_pool2d(depth_image.float(), kernel_size=block_size, stride=block_size, divisor_override=1)

    # Prevent division by zero: if no valid pixels, set count to 1 to avoid NaNs
    valid_counts = torch.where(valid_counts > 0, valid_counts, torch.tensor(1.0, device=depth_image.device))

    # Compute block-wise average
    depth_image_averaged = summed / valid_counts

    # Remove batch & channel dims → (H', W')
    return depth_image_averaged.squeeze(0).squeeze(0)'''

'''def block_averaging(depth_image, block_size, sigma_range, device):
    # Get the dimensions of the depth image
    height, width = depth_image.shape

    # Create an empty tensor to store the averaged depth image
    depth_image_averaged = torch.zeros((height // block_size, width // block_size), dtype=torch.float32).to(device)

    # Loop over the image in blocks
    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            # Get the current block (2x2)
            block = depth_image[y:y + block_size, x:x + block_size]

            # Mask out invalid (zero) depth values in the block
            valid_values = block[block > 0]

            # If there are valid values, compute the average
            if valid_values.numel() > 0:
                avg_depth = valid_values.mean()
            else:
                avg_depth = 0

            # Store the average in the corresponding position of the averaged image
            depth_image_averaged[y // block_size, x // block_size] = avg_depth

    # Return the averaged depth image
    return depth_image_averaged'''


# MUCH FASTER but crashes on 2nd while loop :c
'''import pycuda.driver as cuda
import pycuda.autoinit  # This is needed to initialize the CUDA driver
import numpy as np
from pycuda.compiler import SourceModule

# Kernel function
kernel_code = """
__global__ void BlockAveragingAndSubsampling(
    const uint16_t* depthImage,
    uint16_t* depthImageAveraged,
    const unsigned int width,
    const unsigned int height,
    const unsigned int sigmaRange,
    const unsigned int blockSize)
{
    // Global coordinates of the current thread in lower dim image
    const unsigned int globalX = threadIdx.x + blockIdx.x * blockDim.x;
    const unsigned int globalY = threadIdx.y + blockIdx.y * blockDim.y;

    // Skip if we're out of bounds
    if (globalX >= width - 1 || globalY >= height - 1)
    {
        return;
    }

    // Calculate the similar depth range threshold based on sigmaRange
    const int similarDepthRange = 3 * sigmaRange;  

    const unsigned int globalX_original = globalX * 2;
    const unsigned int globalY_original = globalY * 2;
    const unsigned int width_original = width * 2;

    // Get the 2x2 block of pixels centered on (globalX, globalY)
    const uint16_t topLeftValue = depthImage[globalY_original * width_original + globalX_original];
    const uint16_t topRightValue = depthImage[globalY_original * width_original + globalX_original + 1];
    const uint16_t bottomLeftValue = depthImage[(globalY_original + 1) * width_original + globalX_original];
    const uint16_t bottomRightValue = depthImage[(globalY_original + 1) * width_original + globalX_original + 1];

    unsigned int sum = 0;
    float norm = 0.f;

    // Ignore invalid depth values
    if (topLeftValue > 0) {
        sum += topLeftValue;
        norm++;
    }
    if (topRightValue > 0 && abs(topRightValue - topLeftValue) <= similarDepthRange) {
        sum += topRightValue;
        norm++;
    }
    if (bottomLeftValue > 0 && abs(bottomLeftValue - topLeftValue) <= similarDepthRange) {
        sum += bottomLeftValue;
        norm++;
    }
    if (bottomRightValue > 0 && abs(bottomRightValue - topLeftValue) <= similarDepthRange) {
        sum += bottomRightValue;
        norm++;
    }

    // Calculate the average of the block
    const uint16_t average = (norm > 0) ? uint16_t(float(sum) / norm) : 0;

    // Write the averaged value to the downsampled image (each thread writes to every second pixel)
    depthImageAveraged[globalY * width + globalX] = average;
}
"""

# Compile the kernel code
mod = SourceModule(kernel_code)
block_averaging_kernel = mod.get_function("BlockAveragingAndSubsampling")

# Define the function to launch the kernel
def launch_block_averaging_and_subsample_kernel(depth_frame_data, width, height, block_size, sigma_range):
    # Prepare input image and output image
    depth_frame_data = np.asarray(depth_frame_data, dtype=np.uint16)  # The depth data is a flat 1D array
    depth_image_averaged = np.zeros((height//2, width//2), dtype=np.uint16)  # Averaged depth image

    # Allocate device memory
    d_depth_image = cuda.mem_alloc(depth_frame_data.nbytes)
    d_depth_image_averaged = cuda.mem_alloc(depth_image_averaged.nbytes)

    # Copy data from host to device
    cuda.memcpy_htod(d_depth_image, depth_frame_data)
    cuda.memcpy_htod(d_depth_image_averaged, depth_image_averaged)

    # Define block and grid size
    threads_per_block = (16, 16, 1)

    print(f"Width: {width}, Height: {height}")
    print(f"Threads per block: {threads_per_block}")
    print(f"Width // 2: {width // 2}, Height // 2: {height // 2}")

    num_blocks = ((width // 2 + threads_per_block[0] - 1) // threads_per_block[0],
                  (height // 2 + threads_per_block[1] - 1) // threads_per_block[1])

    print(f"Kernel reference: {block_averaging_kernel}")

    # Launch kernel
    block_averaging_kernel(
        d_depth_image, d_depth_image_averaged, np.uint32(width//2), np.uint32(height//2),
        np.uint32(sigma_range), np.uint32(block_size),
        block=threads_per_block, grid=num_blocks
    )

    # Copy the result back to the host
    cuda.memcpy_dtoh(depth_image_averaged, d_depth_image_averaged)

    # Free device memory
    d_depth_image.free()
    d_depth_image_averaged.free()

    return depth_image_averaged
'''
