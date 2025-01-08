import os
import numpy as np
import torch
import imageio.v3 as iio
import open3d as o3d

from icp.levenberg_marquardt import LM_optimizer
from icp.icp import ICP


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
device = torch.device("cpu")

# define allowed range of depth values in meters
d_max = 3
d_min = 0.25

fx, fy, cx, cy = 517.3, 516.5, 318.6, 255.3
K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]).to(device)


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
print(len(data), data[:20])

optimizer = LM_optimizer()
icp = ICP(max_iterations=10, optimizer=optimizer, symmetric_error=True)

depth, rgb, c2w = read_data(data, 0)

depth1, rgb1, c2w1 = read_data(data, 1)
T10 = icp(depth1, depth, torch.eye(4).to(device), K)

res = c2w @ T10
print(c2w1, res, sep="\n")

H, W = depth.shape

vertices = ICP.compute_vertices(depth, K)
normals = ICP.compute_normals(vertices)
colors = np.array([1, 0, 0]).reshape(1, 3).repeat(H*W, axis=0)
pcd = plot_3d_figure(vertices, normals, colors)

vertices1 = ICP.compute_vertices(depth1, K)
normals1 = ICP.compute_normals(vertices1)
colors1 = np.array([0, 0, 1]).reshape(1, 3).repeat(H*W, axis=0)

R = T10[:3, :3]
t = T10[:3, -1]
vertices1 = torch.matmul(vertices1, R.T) + t

pcd1 = plot_3d_figure(vertices1, normals1, colors1)

o3d.visualization.draw_geometries([pcd, pcd1], point_show_normal=False)
