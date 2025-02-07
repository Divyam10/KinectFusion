import torch

from icp.levenberg_marquardt import LM_optimizer
from icp.icp import ICP
from volume_ray_final import *
from primesense import openni2
import threading
import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
# from bilateral_filter import launch_bilateral_filtering_kernel
from block_averaging_subsampling import block_averaging

is_running = threading.Event()
cuda_device = None
optimizer = LM_optimizer(max_iterations=5)
icp = ICP(optimizer=optimizer, occlusion_threshold=1*1000, symmetric_error=True)


def device_init():
    # Constants
    height = 480
    width = 640
    height_l2 = height // 2
    width_l2 = width // 2
    height_l3 = height_l2 // 2
    width_l3 = width_l2 // 2
    fps = 30

    openni2.initialize()  # can also accept the path of the OpenNI redistribution

    dev = openni2.Device.open_any()
    dev.set_depth_color_sync_enabled(True)

    depth_stream = dev.create_depth_stream()
    color_stream = dev.create_color_stream()

    depth_stream.configure_mode(width, height, fps, openni2.PIXEL_FORMAT_DEPTH_1_MM)
    color_stream.configure_mode(width, height, fps, openni2.PIXEL_FORMAT_RGB888)

    depth_stream.start()
    color_stream.start()

    dev.set_image_registration_mode(openni2.IMAGE_REGISTRATION_DEPTH_TO_COLOR)

    depth_max = depth_stream.get_max_pixel_value()
    depth_min = depth_stream.get_min_pixel_value()

    fx = (0.5 * width) // np.tan(0.5 * depth_stream.get_horizontal_fov())
    fy = (0.5 * height) // np.tan(0.5 * depth_stream.get_vertical_fov())
    px = width // 2.0
    py = height // 2.0

    k1 = np.array([
        [fx, 0.0, px],
        [0.0, fy, py],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)
    k2 = np.array([
        [0.5 * fx, 0.0, 0.5 * px],
        [0.0, 0.5 * fy, 0.5 * py],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)
    k3 = np.array([
        [0.25 * fx, 0.0, 0.25 * px],
        [0.0, 0.25 * fy, 0.25 * py],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)
    k_pyr = [k1, k2, k3]
    return dev, depth_stream, color_stream, depth_min, depth_max, k_pyr, width, height, width_l2, width_l3, height_l2, height_l3


def calc_icp(
        depth_pyr: list,
        tsdf_depth_pyr: list,
        k_pyr: list,
        c2w: np.ndarray
):
    global icp, cuda_device
    t10 = icp(depth_pyr[2].to(torch.float32),
              tsdf_depth_pyr[2],
              torch.tensor(np.identity(4), dtype=torch.float32).to(cuda_device),
              torch.tensor(k_pyr[2], dtype=torch.float32).to(cuda_device))
    print("icp l3")
    print(t10)
    t10 = icp(depth_pyr[1].to(torch.float32),
              tsdf_depth_pyr[1],
              t10,
              torch.tensor(k_pyr[1], dtype=torch.float32).to(cuda_device))
    print("icp l2")
    print(t10)
    t10 = icp(depth_pyr[0].to(torch.float32),
              tsdf_depth_pyr[0],
              t10,
              torch.tensor(k_pyr[0], dtype=torch.float32).to(cuda_device))
    print("icp l1")
    print(t10)
    # TODO range/instead check for Null?
    if torch.allclose(t10, torch.eye(4).to(cuda_device), atol=1e-3):  #TODO
        print("ICP failed or did not improve pose")
    else:
        c2w = c2w @ t10.cpu().numpy()
    #torch.cuda.synchronize()  # TODO maybe not necessary
    #c2w = c2w @ t10.cpu().numpy()
    print("c2w")
    print(c2w)
    return c2w


def process_frames(depth_stream, color_stream, depth_min, depth_max, k_pyr, width, height, width_l2, width_l3,
                   height_l2, height_l3, sigma_spatial, sigma_range):
    global is_running, cuda_device

    c2w = np.eye(4)
    volume_bounds = None
    vox_grid = None
    last_frame = None
    iteration = 0 # TODO remove
    while is_running.is_set():
        depth_frame = depth_stream.read_frame()
        color_frame = color_stream.read_frame()

        depth_frame_data = torch.frombuffer(depth_frame.get_buffer_as_uint16(), dtype=torch.uint16).reshape((height, width))
        depth_frame_data = depth_frame_data.to(torch.float32).to(cuda_device)
        color_map = torch.frombuffer(color_frame.get_buffer_as_uint8(), dtype=torch.uint8).reshape((height, width, 3))

        # TODO
        '''depth_map_l1, validity_mask = launch_bilateral_filtering_kernel(
            depth_frame_data, sigma_spatial, sigma_range, depth_min, depth_max, width, height
        )'''
        depth_map_l2 = block_averaging(
            depth_frame_data, 2, sigma_range
        )
        depth_map_l3 = block_averaging(
            depth_map_l2, 2, sigma_range
        )

        depth_map_l1 = depth_frame_data.to(torch.uint16)
        depth_map_l2 = depth_map_l2.to(torch.uint16)
        depth_map_l3 = depth_map_l3.to(torch.uint16)
        dep_pyr = [depth_map_l1, depth_map_l2, depth_map_l3]

        current_frame = [color_map, dep_pyr]

        if last_frame is None:
            print("First Frame...")
            print("Computing volume bounds...")
            volume_bounds = get_vol_bnds(depth_frame_data, k_pyr[0], c2w)
            print("Computing voxel grid...")
            vox_grid = TSDF(vol_dim=volume_bounds, intristics=k_pyr[0])
            print("Voxel grid... Done!")
            last_frame = current_frame
            continue

        # TODO ? depth_frame[depth_frame == 65535] = 0

        tsdf_dep_pyr, rgb_pyr, vtx_pyr, nrm_pyr, mask_pyr = vox_grid.render_pyramid(
            c2w=c2w,
            intri=k_pyr[0],
            imh=height,
            imw=width,
            n_pyr=3
        )

        c2w = calc_icp(dep_pyr, tsdf_dep_pyr, k_pyr, c2w)

        vox_grid.integrate(current_frame[1][0], c2w, current_frame[0])

        iteration += 1
        if iteration == 50:
            get_mesh(vox_grid)
            print("Mesh generation... Done!")
            return

        '''        plt.figure(figsize=(10, 10))

        # Plot L1 (Original + Bilateral Filtered Depth Map)
        plt.subplot(1, 3, 1)
        plt.imshow(depth_frame_data.cpu().numpy(), cmap='gray')
        plt.title("Depth Map L1")
        plt.axis('off')

        # Plot L2 (Subsampled Depth Map)
        plt.subplot(1, 3, 2)
        plt.imshow(depth_map_l2.cpu().numpy(), cmap='gray')
        plt.title("Depth Map L2")
        plt.axis('off')

        # Plot L3 (Further Subsampled Depth Map)
        plt.subplot(1, 3, 3)
        plt.imshow(depth_map_l3.cpu().numpy(), cmap='gray')
        plt.title("Depth Map L3")
        plt.axis('off')

        # Display the plots
        plt.show()'''

        # Reset frames:
        depth_frame = None
        color_frame = None
        continue

        # print("Preprocessing... done!\n")


def shutdown(dev, depth_stream, color_stream):
    depth_stream.stop()
    color_stream.stop()
    openni2.unload()


def main():
    global is_running, cuda_device

    if torch.cuda.is_available():
        print("Using CUDA")
        cuda_device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        print("Using MPS")
        cuda_device = torch.device("mps")
    else:
        print("Using CPU")
        cuda_device = torch.device("cpu")

    sigma_spatial = 10
    sigma_range = 100

    dev, depth_stream, color_stream, depth_min, depth_max, k_pyr, width, height, width_l2, width_l3, height_l2, height_l3 = device_init()
    is_running.set()
    process_frames(depth_stream, color_stream, depth_min, depth_max, k_pyr, width, height, width_l2, width_l3,
                   height_l2, height_l3, sigma_spatial, sigma_range)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
