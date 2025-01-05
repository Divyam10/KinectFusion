import os
import numpy as np
import torch
import imageio.v3 as iio


DATA_DIR = 'data/rgbd_dataset_freiburg1_desk'
DATA_PATH = os.path.join(os.getcwd(), DATA_DIR)

depth_file = os.path.join(DATA_PATH, 'depth.txt')
rgb_file = os.path.join(DATA_PATH, 'rgb.txt')
trajectory_file = os.path.join(DATA_PATH, 'groundtruth.txt')

# define allowed range of depth values in meters
d_max = 3
d_min = 0.25

fx, fy, cx, cy = 517.3, 516.5, 318.6, 255.3
K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])


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


data = prepare_data(depth_file, rgb_file, trajectory_file)
print(len(data), data[:20])

depth = iio.imread(os.path.join(DATA_PATH, data[0]["data"][0])).astype(np.float32)

# Rescale depth to meters (https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats#intrinsic_camera_calibration_of_the_kinect)
depth /= 5000
print(depth.max(), depth.min())
depth[(depth < d_min) | (depth > d_max)] = 0
print(type(depth), depth.shape, depth.dtype)

rgb = iio.imread(os.path.join(DATA_PATH, data[0]["data"][8])).astype(np.float32)
print(type(rgb), rgb.shape, rgb.dtype)


depth = torch.from_numpy(depth)

def compute_vertices(depth_map, K):
    H, W = depth_map.shape
    device = depth_map.device
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    pixel_grid_v, pixel_grid_u = torch.meshgrid([torch.arange(0, W), torch.arange(0, H)], indexing="xy")
    pixel_grid_u.to(device)
    pixel_grid_v.to(device)

    print(pixel_grid_u.shape, pixel_grid_v.shape)

    pixel_grid_u = ((pixel_grid_u - cx) / fx) * depth_map
    pixel_grid_v = ((pixel_grid_v - cy) / fy) * depth_map

    vertices = torch.stack((pixel_grid_u, pixel_grid_v, depth_map), dim=2)
    return vertices

vertices = compute_vertices(depth, K)
print(vertices.shape, vertices.dtype)
print(vertices[155, 155, :])
