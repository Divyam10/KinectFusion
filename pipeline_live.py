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


DATA_DIR = '/home/zeus/masters/3DSMC/project/KinectFusion_/dataset/rgbd_dataset_freiburg1_desk'
DATA_PATH = os.path.join(os.getcwd(), DATA_DIR)

depth_file = os.path.join(DATA_PATH, 'depth.txt')
rgb_file = os.path.join(DATA_PATH, 'rgb.txt')
trajectory_file = os.path.join(DATA_PATH, 'groundtruth.txt')

height = 480
width = 640
height_l2 = height // 2
width_l2 = width // 2
height_l3 = height_l2 // 2
width_l3 = width_l2 // 2
fps = 30

dist = "/home/zeus/Install/kinect/openni2/OpenNI2/Packaging/OpenNI2-x64/Redist/"
# can also accept the path of the OpenNI redistribution
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
# device = torch.device("cpu")

# define allowed range of depth values in meters
d_max = 5
d_min = 0.25

num_scales = 3

K = torch.tensor([[fx, 0, px], [0, fy, py], [0, 0, 1]]).to(dtype=torch.float64).to(device)
c2w = torch.eye(4, dtype=torch.float64, device=device)

c2w[0, 3] = -0.25  # -0.25
c2w[1, 3] = 1.0  # 1.0
c2w[2, 3] = -0.1

def get_time():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif torch.backends.mps.is_available():
        torch.mps.synchronize()

    return time.time()


def plot_3d_figure(point_cloud, normals, colors_np):
    point_cloud_np = point_cloud.view(-1, 3).cpu().numpy()
    normals_np = normals.view(-1, 3).cpu().numpy()
    # colors_np = colors_np.view(-1, 3).cpu().numpy()

    # Normalize colors (ensure they are in range [0, 1])
    if colors_np.max() > 1.0:
        colors_np = colors_np / 255.0

    pcd = o3d.geometry.PointCloud()

    pcd.points = o3d.utility.Vector3dVector(point_cloud_np)
    pcd.normals = o3d.utility.Vector3dVector(normals_np)
    pcd.colors = o3d.utility.Vector3dVector(colors_np)

    return pcd


optimizer = LM_optimizer(max_iterations=10)
icp = ICP(optimizer=None, occlusion_threshold=0.1, symmetric_error=True)

icp_solvers = [
    ICP(optimizer=LM_optimizer(max_iterations=6, damping_factor=1.0e-4),
        occlusion_threshold=0.1, symmetric_error=True),
    ICP(optimizer=LM_optimizer(max_iterations=3, damping_factor=1.0e-4),
        occlusion_threshold=0.1, symmetric_error=True),
    ICP(optimizer=LM_optimizer(max_iterations=3, damping_factor=1.0e-2),
        occlusion_threshold=0.1, symmetric_error=True)
]

# icp_solvers = [
#     ICP(optimizer=None, occlusion_threshold=0.1, symmetric_error=True),
#     ICP(optimizer=None, occlusion_threshold=0.1, symmetric_error=True),
#     ICP(optimizer=None, occlusion_threshold=0.1, symmetric_error=True)
# ]

multiscales = [torch.nn.MaxPool2d(1 << i, 1 << i) for i in range(num_scales)]

start = 0
end = 300

# depth, rgb, c2w = read_data(data, start)
done = False
i = 0
while not done:
    depth_frame = depth_stream.read_frame()
    color_frame = color_stream.read_frame()

    depth_frame_data = np.frombuffer(
        depth_frame.get_buffer_as_uint16(), dtype=np.uint16).reshape((height, width))
    depth0 = depth_frame_data.astype(np.float32)
    color0 = np.fromstring(color_frame.get_buffer_as_uint8(),
                           dtype=np.uint8).reshape(480, 640, 3)
    # color0  = cv2.cvtColor(color0,cv2.COLOR_BGR2RGB)
    depth0 /= 1000.
    depth0[(depth0 < 0.1) | (depth0 > 5.0)] = 0.0
    
    color0 = torch.from_numpy(color0).to(device)
    depth0 = torch.from_numpy(depth0).to(device)
    H, W = depth0.shape
    if i == 0:
        volume_bounds = tsdf.get_vol_bnds(
            depth0, K.cpu().numpy(), c2w.cpu().numpy())
        vox_grid = tsdf.TSDF(vol_dim=volume_bounds, intristics=K)
        vox_grid.integrate(depth0, c2w.cpu().numpy(), color0)
        i += 1
        continue

    R = c2w[:3, :3]
    t = c2w[:3, -1]
    # vertices = torch.matmul(vertices, R.T) + t
    # pcd = plot_3d_figure(vertices, normals, colors)

    time_list, c2w_list, c2w_gt_list = list(), list(), list()
    # depth_curr, rgb_curr, c2w_curr = read_data(data, i)
    t0 = get_time()

    depth1, color1, vertex01, normal1, mask1 = vox_grid.render_model(c2w.cpu().numpy(), K, H, W, near=d_min,
                                                                     far=d_max, n_samples=192)

    dpt_curr_pyr = [f(depth0.view(1, 1, H, W)) for f in multiscales]
    dpt_curr_pyr = [d.squeeze() for d in dpt_curr_pyr]
    dpt1_pyr = [f(depth1.view(1, 1, H, W)) for f in multiscales]
    dpt1_pyr = [d.squeeze() for d in dpt1_pyr]

    T10 = torch.eye(4, dtype=torch.float64,).to(device)

    for j in reversed(range(num_scales)):
        K_scaled = K.clone()
        if j != 0:
            K_scaled[0, 0] /= 2 ** j
            K_scaled[1, 1] /= 2 ** j
            K_scaled[0, 2] /= 2 ** j
            K_scaled[1, 2] /= 2 ** j

        T10, err_msg = icp_solvers[j](
            dpt_curr_pyr[j], dpt1_pyr[j], T10, K_scaled)
        if err_msg:
            print("ERROR:", err_msg)
        else:
            print("No error")
        # print("T10 -", j)
        # print(T10)

    c2w = c2w @ T10

    vox_grid.integrate(depth0, c2w.cpu().numpy(), color0)

    t1 = get_time()
    time_list += [t1 - t0]
    # print("processed frame: {:d}, time taken: {:f}s".format(i, t1 - t0))
    i+=1
    if i==300:
        break


avg_time = np.array(time_list).mean()
print("average processing time: {:f}s per frame, i.e. {:f} fps".format(
    avg_time, 1. / avg_time))

tsdf.get_mesh(vox_grid)
print("Mesh generation... Done!")

# c2w_gt_list = np.stack(c2w_gt_list, 0)
c2w_list = np.stack(c2w_list, 0)
# traj_gt = np.array(c2w_gt_list)[:, :3, 3]
traj = np.array(c2w_list)[:, :3, 3]
# print(c2w_gt_list[-1])
print(c2w_list[-1])
plt.figure(figsize=(10, 6))
# plt.plot(traj_gt[:, 0], traj_gt[:, 1], label="Ground Truth", color="blue")
plt.plot(traj[:, 0], traj[:, 1], label="Estimated",
         color="red", linestyle="--")
plt.legend()
plt.title("Trajectory Comparison")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid()
plt.show()

# rmse = np.sqrt(np.mean(np.linalg.norm(traj_gt - traj, axis=-1) ** 2))
# print("RMSE: {:f}".format(rmse))
#
#
# vertices1 = ICP.compute_vertices(depth1, K)
# normals1 = ICP.compute_normals(vertices1)
# colors1 = np.array([0, 0, 1]).reshape(1, 3).repeat(H*W, axis=0)
#
# R = c2w[:3, :3]
# t = c2w[:3, -1]
# vertices1 = torch.matmul(vertices1, R.T) + t
# pcd1 = plot_3d_figure(vertices1, normals1, colors1)
#
# o3d.visualization.draw_geometries([pcd, pcd1], point_show_normal=False)
