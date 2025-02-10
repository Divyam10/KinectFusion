import os
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import imageio.v3 as iio
import open3d as o3d

from icp.levenberg_marquardt import LM_optimizer
from icp.icp import ICP

import volume_ray_final as tsdf
from primesense import openni2
import cv2

DATA_DIR = '/home/zeus/masters/3DSMC/project/KinectFusion_/dataset/rgbd_dataset_freiburg1_desk'
DATA_PATH = os.path.join(os.getcwd(), DATA_DIR)

depth_file = os.path.join(DATA_PATH, 'depth.txt')
rgb_file = os.path.join(DATA_PATH, 'rgb.txt')
trajectory_file = os.path.join(DATA_PATH, 'groundtruth.txt')

height = 480//2
width = 640//2
height_l2 = height // 2
width_l2 = width // 2
height_l3 = height_l2 // 2
width_l3 = width_l2 // 2
fps = 30

dist = "/home/zeus/Install/kinect/openni2/OpenNI2/Packaging/OpenNI2-x64/Redist/"
openni2.initialize(dist)

dev = openni2.Device.open_any()
dev.set_depth_color_sync_enabled(True)

depth_stream = dev.create_depth_stream()
color_stream = dev.create_color_stream()

depth_stream.configure_mode(
    width, height, fps, openni2.PIXEL_FORMAT_DEPTH_1_MM)
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


if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

d_max = 5 #TODO check
d_min = 0.25

num_scales = 3

K = torch.tensor([[fx, 0, px], [0, fy, py], [0, 0, 1]]
                 ).to(dtype=torch.float32).to(device)
c2w = torch.eye(4, dtype=torch.float32, device=device)

c2w[0, 3] = -0.25
c2w[1, 3] = 1.0
c2w[2, 3] = -0.1


def get_time():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif torch.backends.mps.is_available():
        torch.mps.synchronize()

    return time.time()





optimizer = LM_optimizer(max_iterations=10)
icp = ICP(optimizer=None, occlusion_threshold=0.1, symmetric_error=True)

'''icp_solvers = [
    ICP(optimizer=None,
        occlusion_threshold=0.1, symmetric_error=True),
    ICP(optimizer=None,
        occlusion_threshold=0.1, symmetric_error=True),
    ICP(optimizer=None,
        occlusion_threshold=0.1, symmetric_error=True)
]'''

icp_solvers = [
    ICP(optimizer=LM_optimizer(max_iterations=6, damping_factor=1.0e-4),
        occlusion_threshold=0.1, symmetric_error=True),
    ICP(optimizer=LM_optimizer(max_iterations=3, damping_factor=1.0e-4),
        occlusion_threshold=0.1, symmetric_error=True),
    ICP(optimizer=LM_optimizer(max_iterations=3, damping_factor=1.0e-2),
        occlusion_threshold=0.1, symmetric_error=True)
]

'''icp_solvers = [
    ICP(optimizer=None,
        occlusion_threshold=0.1, symmetric_error=True),
    ICP(optimizer=None,
        occlusion_threshold=0.1, symmetric_error=True),
    ICP(optimizer=LM_optimizer(max_iterations=3, damping_factor=1.0e-2),
        occlusion_threshold=0.1, symmetric_error=True)
]'''

'''icp_solvers = [
    ICP(optimizer=LM_optimizer(max_iterations=3, damping_factor=1.0e-4),
        occlusion_threshold=0.1, symmetric_error=True),
    ICP(optimizer=None,
        occlusion_threshold=0.1, symmetric_error=True),
    ICP(optimizer=None,
        occlusion_threshold=0.1, symmetric_error=True)
]'''


multiscales = [torch.nn.MaxPool2d(1 << i, 1 << i) for i in range(num_scales)]

start = 0
end = 300

done = False
i = 0

j = 0
while not done:
    t_data_loading = get_time()
    #t0 = get_time()
    depth_frame = depth_stream.read_frame()
    color_frame = color_stream.read_frame()
    
    if j<=15:
        j+=1
        continue

    #TODO
    depth0 = np.frombuffer(depth_frame.get_buffer_as_uint16(), dtype=np.uint16)
    depth0 = torch.from_numpy(depth0.astype(np.float32)).to(device).reshape((height, width))
    #depth0 = torch.frombuffer(depth_frame.get_buffer_as(ctype=openni2.ctypes.c_uint16), dtype=torch.int16).to(device).reshape((height, width))
    depth0 = torch.flip(depth0, dims=[1, 0])
    #depth0 = depth0.to(torch.float32)
    
    color0 = np.frombuffer(color_frame.get_buffer_as_uint8(), dtype=np.uint8)
    color0 = torch.from_numpy(color0).to(device).reshape(height, width, 3)
    #color0 = torch.frombuffer(color_frame.get_buffer_as(ctype=openni2.ctypes.c_uint8), dtype=torch.int8).to(device).reshape(480, 640, 3)
    color0 = torch.flip(color0, dims=[1, 0])
    color0 = color0[:, :, [2, 1, 0]]  # Swap channels from BGR to RGB
    #color0 = color0.to(torch.float32)
    
    depth0 /= 1000.
    depth0[(depth0 < 0.1) | (depth0 > 5.0)] = 0.0

    H, W = depth0.shape #TODO remove
    
   
    
    
    if i == 0:
        volume_bounds = tsdf.get_vol_bnds(
            depth0, K.cpu().numpy(), c2w.cpu().numpy())
        # volume_bounds = np.array([[-2.80372819,  2.30372819],
        #                  [-2.16529614,  2.16529614],
        #                  [-2.1,         2.68699994]])
        vox_grid = tsdf.TSDF(vol_dim=volume_bounds, intristics=K)
        vox_grid.integrate(depth0, c2w, color0)
        i += 1
        continue

    time_list, c2w_list, c2w_gt_list = list(), list(), list()
   
    t_tsdf = get_time()

    depth1, color1, vertex01, normal1, mask1 = vox_grid.render_model(c2w, K, H, W, near=d_min,
                                                                     far=d_max, n_samples=192)
    
    t_tsdf_done = get_time()

    dpt_curr_pyr = [f(depth0.view(1, 1, H, W)) for f in multiscales]
    dpt_curr_pyr = [d.squeeze() for d in dpt_curr_pyr]
    dpt1_pyr = [f(depth1.view(1, 1, H, W)) for f in multiscales]
    dpt1_pyr = [d.squeeze() for d in dpt1_pyr]

    t_icp = get_time()

    T10 = torch.eye(4, dtype=torch.float32,).to(device)

    err_msgs = ""
    try:
        for j in reversed(range(num_scales)):
            K_scaled = K.clone()
            if j != 0:
                K_scaled[0, 0] /= 2 ** j
                K_scaled[1, 1] /= 2 ** j
                K_scaled[0, 2] /= 2 ** j
                K_scaled[1, 2] /= 2 ** j

            T10, err_msg = icp_solvers[j](
                dpt_curr_pyr[j], dpt1_pyr[j], T10, K_scaled)

            err_msgs += err_msg + "\n"

        if err_msg:
            print("ERROR:", err_msgs)
            print("Skipping current frame...")
        else:
            print("No errors found, integrating current frame...")
            c2w = c2w @ T10
            vox_grid.integrate(depth0, c2w, color0)
            # print("T10 -", j)
            # print(T10)
    except Exception as X:
        print(X)

    t_icp_done = get_time()
    #t1 = get_time()
    #time_list += [t1 - t0]
    i += 1
    # print("processed frame: {:d}, time data: {:f}s".format(i, t_data_loading_done - t_data_loading))
    # print("processed frame: {:d}, time tsdf: {:f}s".format(i, t_tsdf_done - t_tsdf))
    # print("processed frame: {:d}, time icp: {:f}s".format(i, t_icp_done - t_icp))
    # print("processed frame: {:d}, time taken: {:f}s".format(i, t1 - t0))
    print(1/(time.time() - t_data_loading))
    if i == 50:
        break


avg_time = np.array(time_list).mean()
print("average processing time: {:f}s per frame, i.e. {:f} fps".format(
    avg_time, 1. / avg_time))

tsdf.get_mesh(vox_grid)
print("Mesh generation... Done!")

