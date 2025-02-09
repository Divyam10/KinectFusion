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


DATA_DIR = 'data/rgbd_dataset_freiburg1_desk'
DATA_PATH = os.path.join(os.getcwd(), DATA_DIR)

depth_file = os.path.join(DATA_PATH, 'depth.txt')
rgb_file = os.path.join(DATA_PATH, 'rgb.txt')
trajectory_file = os.path.join(DATA_PATH, 'groundtruth.txt')

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

fx, fy, cx, cy = 517.3, 516.5, 318.6, 255.3
K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]).to(device)


def get_time():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif torch.backends.mps.is_available():
        torch.mps.synchronize()

    return time.time()


def read_data_file(file):
    data = []
    with open(file, 'r') as f:
        for line in f.readlines():
            if line and line[0] == "#":
                continue

            line = line.split()
            data.append({"timestamp": float(line[0]), "data": line[1:]})

    return data


def align_data(data_a, data_b, max_diff=0.25):
    min_diff = float("inf")
    cur_match = None
    for a in data_a:
        for b in data_b:
            cur_diff = abs(a["timestamp"] - b["timestamp"])
            if not cur_match or cur_diff < min_diff:
                min_diff = cur_diff
                cur_match = b["data"]

        if min_diff <= max_diff:
            a["data"].extend(cur_match)
        else:
            a["timestamp"] = -1

    data_a = [x for x in data_a if x["timestamp"] != -1]
    data_a.sort(key=lambda x: x["timestamp"])

    return data_a



def prepare_data(depth_file, rgb_file, trajectory_file):
    depth_data = read_data_file(depth_file)
    rgb_data = read_data_file(rgb_file)
    trajectory_data = read_data_file(trajectory_file)

    combined_data = depth_data
    combined_data = align_data(combined_data, trajectory_data)
    combined_data = align_data(combined_data, rgb_data)

    return combined_data


def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    qw, qx, qy, qz = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (qy * qy + qz * qz),
            two_s * (qx * qy - qz * qw),
            two_s * (qx * qz + qy * qw),
            two_s * (qx * qy + qz * qw),
            1 - two_s * (qx * qx + qz * qz),
            two_s * (qy * qz - qx * qw),
            two_s * (qx * qz - qy * qw),
            two_s * (qy * qz + qx * qw),
            1 - two_s * (qx * qx + qy * qy),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


# def load_K_Rt_from_P(P):
#     """
#     modified from IDR https://github.com/lioryariv/idr
#     """
#     P = P.detach().cpu().numpy()
#     out = cv2.decomposeProjectionMatrix(P)
#     K = out[0]
#     R = out[1]
#     t = out[2]
#
#     K = K/K[2,2]
#     intrinsics = np.eye(4)
#     intrinsics[:3, :3] = K
#
#     pose = np.eye(4, dtype=np.float32)
#     pose[:3, :3] = R.transpose()  # convert from w2c to c2w
#     pose[:3, 3] = (t[:3] / t[3])[:, 0]
#
#     return intrinsics, pose


def read_data(data, index):
    assert index < len(data)

    depth = iio.imread(os.path.join(DATA_PATH, data[index]["data"][0])).astype(np.float32)
    depth /= 5000
    depth[(depth < d_min) | (depth > d_max)] = 0
    depth = torch.from_numpy(depth).to(device)

    rgb = iio.imread(os.path.join(DATA_PATH, data[0]["data"][8])).astype(np.float32)
    rgb = torch.from_numpy(rgb).to(device)

    trajectories = list(map(float, data[index]["data"][1:8]))
    trajectories = torch.tensor(trajectories, dtype=torch.float32).to(device)

    # qx, qy, qz, qw -> qw, qx, qy, qz
    trajectories[[3, 4, 5, 6]] = trajectories[[6, 3, 4, 5]]
    R = quaternion_to_matrix(trajectories[3:])
    c = trajectories[:3]
    # w2c = torch.eye(4).to(device)
    # w2c[:3, :3] = R
    # w2c[:3, 3] = c
    # w2c = torch.linalg.inv(w2c)
    #
    # my_c2w = torch.eye(4).to(device)
    # my_c2w[:3, :3] = R
    # my_c2w[:3, 3] = c

    c2w = torch.eye(4).to(device)
    c2w[:3, :3] = R
    c2w[:3, 3] = c

    return depth, rgb, c2w

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


data = prepare_data(depth_file, rgb_file, trajectory_file)
optimizer = LM_optimizer(max_iterations=10)
icp = ICP(optimizer=None, occlusion_threshold=0.1, symmetric_error=True)

icp_solvers = [
    ICP(optimizer=LM_optimizer(max_iterations=6, damping_factor=1.0e-4), occlusion_threshold=0.1, symmetric_error=True),
    ICP(optimizer=LM_optimizer(max_iterations=3, damping_factor=1.0e-4), occlusion_threshold=0.1, symmetric_error=True),
    ICP(optimizer=LM_optimizer(max_iterations=3, damping_factor=1.0e-2), occlusion_threshold=0.1, symmetric_error=True)
]

# icp_solvers = [
#     ICP(optimizer=None, occlusion_threshold=0.1, symmetric_error=True),
#     ICP(optimizer=None, occlusion_threshold=0.1, symmetric_error=True),
#     ICP(optimizer=None, occlusion_threshold=0.1, symmetric_error=True)
# ]

multiscales = [torch.nn.MaxPool2d(1<<i, 1<<i) for i in range(num_scales)]

start = 0
end = 30
print(len(data))

depth, rgb, c2w = read_data(data, start)
H, W = depth.shape

volume_bounds = tsdf.get_vol_bnds(depth, K.cpu().numpy(), c2w.cpu().numpy())
vox_grid = tsdf.TSDF(vol_dim=volume_bounds, intristics=K, device=device)

vox_grid.integrate(depth, c2w.cpu().numpy(), rgb)

vertices = ICP.compute_vertices(depth, K)
normals = ICP.compute_normals(vertices)
colors = np.array([1, 0, 0]).reshape(1, 3).repeat(H*W, axis=0)

R = c2w[:3, :3]
t = c2w[:3, -1]
vertices = torch.matmul(vertices, R.T) + t
pcd = plot_3d_figure(vertices, normals, colors)

time_list, c2w_list, c2w_gt_list = list(), list(), list()
for i in range(start+1, min(len(data), end+1)):
    depth_curr, rgb_curr, c2w_curr = read_data(data, i)
    t0 = get_time()

    depth1, color1, vertex01, normal1, mask1 = vox_grid.render_model(c2w.cpu().numpy(), K, H, W, near=d_min,
                                                                        far=d_max, n_samples=192)

    dpt_curr_pyr = [f(depth_curr.view(1, 1, H, W)) for f in multiscales]
    dpt_curr_pyr = [d.squeeze() for d in dpt_curr_pyr]
    dpt1_pyr = [f(depth1.view(1, 1, H, W)) for f in multiscales]
    dpt1_pyr = [d.squeeze() for d in dpt1_pyr]

    T10 = torch.eye(4).to(device)
    for j in reversed(range(num_scales)):
        K_scaled = K.clone()
        if j!= 0:
            K_scaled[0, 0] /= 2 ** j
            K_scaled[1, 1] /= 2 ** j
            K_scaled[0, 2] /= 2 ** j
            K_scaled[1, 2] /= 2 ** j

        T10, err_msg = icp_solvers[j](dpt_curr_pyr[j], dpt1_pyr[j], T10, K_scaled)
        if err_msg:
            print("ERROR:", err_msg)
        else:
            print("No error")
        # print("T10 -", j)
        # print(T10)

    c2w = c2w @ T10

    vox_grid.integrate(depth_curr, c2w.cpu().numpy(), rgb_curr)

    t1 = get_time()
    time_list += [t1 - t0]
    print("processed frame: {:d}, time taken: {:f}s".format(i, t1 - t0))

    c2w_list += [c2w.cpu().numpy()]
    c2w_gt_list += [c2w_curr.cpu().numpy()]
    print(c2w_gt_list[-1])
    print(c2w_list[-1])

avg_time = np.array(time_list).mean()
print("average processing time: {:f}s per frame, i.e. {:f} fps".format(avg_time, 1. / avg_time))

tsdf.get_mesh(vox_grid)
print("Mesh generation... Done!")

c2w_gt_list = np.stack(c2w_gt_list, 0)
c2w_list = np.stack(c2w_list, 0)
traj_gt = np.array(c2w_gt_list)[:, :3, 3]
traj = np.array(c2w_list)[:, :3, 3]
print(c2w_gt_list[-1])
print(c2w_list[-1])
plt.figure(figsize=(10, 6))
plt.plot(traj_gt[:, 0], traj_gt[:, 1], label="Ground Truth", color="blue")
plt.plot(traj[:, 0], traj[:, 1], label="Estimated", color="red", linestyle="--")
plt.legend()
plt.title("Trajectory Comparison")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid()
plt.show()

rmse = np.sqrt(np.mean(np.linalg.norm(traj_gt - traj, axis=-1) ** 2))
print("RMSE: {:f}".format(rmse))
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
