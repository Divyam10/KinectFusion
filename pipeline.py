import os
import time
from datetime import datetime

import numpy as np
import torch
import MeasurementModule
import threading

from matplotlib import pyplot as plt

from icp.levenberg_marquardt import LM_optimizer
from icp.icp import ICP

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

lock = threading.Lock()

last_frame = None
d_max = None  # define allowed range of depth values in meters
d_min = None  # define allowed range of depth values in meters
K, K2, K3 = None, None, None
K_tensor_l1 = None
K_tensor_l2 = None
K_tensor_l3 = None
c2w = None


def debugRandomImage(width, height) -> np.ndarray:
    # Generate depth values between 1 and 10,000
    ndarray = np.random.randint(1, 10001, size=(height, width), dtype=np.uint16)

    # Randomly set some values to 0 (invalid depth readings)
    num_invalid = int(0.1 * width * height)  # 10% invalid values
    invalid_indices = np.random.choice(width * height, num_invalid, replace=False)
    ndarray.flat[invalid_indices] = 0
    return ndarray / 10000


import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


def visualize_shift(last_frame, current_frame, suffix=""):
    # Validate that frames are not None or empty
    if last_frame is None or current_frame is None:
        print("Error: One of the frames is None.")
        return

    if last_frame.size == 0 or current_frame.size == 0:
        print("Error: One of the frames is empty.")
        return

    # Use a non-interactive backend to avoid GUI issues
    plt.switch_backend('Agg')  # 'Agg' is a non-GUI backend suitable for saving images

    print("Creating the visualization...")  # Debugging print

    # Visualize the last frame and current frame side by side
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Last frame
    axes[0].imshow(last_frame, cmap='gray')
    axes[0].set_title('Last Frame')
    axes[0].axis('off')

    # Current frame after shift
    axes[1].imshow(current_frame, cmap='gray')
    axes[1].set_title('Current Frame (Shifted)')
    axes[1].axis('off')

    # Difference frame (how much the pixels have shifted)
    diff_frame = np.abs(current_frame - last_frame)
    axes[2].imshow(diff_frame, cmap='hot')
    axes[2].set_title('Difference (Shift Visualization)')
    axes[2].axis('off')

    plt.tight_layout()
    plt.suptitle('Simulated Depth Map')

    # Get current time for filename
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Save the visualization to a file
    save_dir = "C:/Users/steph/Documents/Projekte/KinectFusion/"
    if not os.path.exists(save_dir):
        print(f"Error: The directory {save_dir} does not exist.")
        return

    full_filename = os.path.join(save_dir, f"diff_map_{current_time}{suffix}.png")

    try:
        print(f"Saving visualization to {full_filename}")  # Debugging print
        plt.savefig(full_filename)
        print(f"Visualization saved successfully at {full_filename}")  # Success message
    except Exception as e:
        print(f"Error saving image: {e}")

    # Close the plot to avoid memory issues
    plt.close()


def on_new_frame():
    global last_frame, K_tensor_l1, K_tensor_l2, K_tensor_l3, d_max, d_min, c2w
    print("called\n")

    with lock:
        current_frame = None

        if last_frame is None:
            last_frame = MeasurementModule.PopFrame()

            # Assume World coordinates align with first frame camera coord system.
            c2w = torch.tensor(np.eye(4), dtype=torch.float32).to(device)
            K_tensor_l1 = torch.tensor(MeasurementModule.Device.K()).to(device)
            K_tensor_l2 = torch.tensor(MeasurementModule.Device.K2()).to(device)
            K_tensor_l3 = torch.tensor(MeasurementModule.Device.K3()).to(device)

            # d_max = MeasurementModule.Device.maxDepth()
            d_max = 10000
            # d_min = MeasurementModule.Device.minDepth()
            d_min = 0

            MeasurementModule.Device.set_python_processing(False)
            return

        current_frame = MeasurementModule.PopFrame()

        T10 = icp(torch.tensor(current_frame.l3.depth_map, dtype=torch.float32).to(device),
                  torch.tensor(last_frame.l3.depth_map, dtype=torch.float32).to(device),
                  torch.eye(4).to(device),
                  K_tensor_l3)

        T10 = icp(torch.tensor(current_frame.l2.depth_map, dtype=torch.float32).to(device),
                  torch.tensor(last_frame.l2.depth_map, dtype=torch.float32).to(device),
                  T10,
                  K_tensor_l2)

        T10 = icp(torch.tensor(current_frame.l1.depth_map, dtype=torch.float32).to(device),
                  torch.tensor(last_frame.l1.depth_map, dtype=torch.float32).to(device),
                  T10,
                  K_tensor_l1)


        #visualize_shift(last_frame.raw_depth, current_frame.raw_depth, "raw")
        #visualize_shift(last_frame.l1.depth_map, current_frame.l1.depth_map, "l1")
        #visualize_shift(last_frame.l2.depth_map, current_frame.l2.depth_map, "l2")
        #visualize_shift(last_frame.l3.depth_map, current_frame.l3.depth_map, "l3")

        c2w = c2w @ T10
        print("C2W!", c2w)

        MeasurementModule.Device.set_python_processing(False)


optimizer = LM_optimizer(max_iterations=5)
icp = ICP(optimizer=None, occlusion_threshold=1, symmetric_error=True)

# Starts Program
MeasurementModule.Init(on_new_frame)
# on_new_frame()  # TODO remove
# on_new_frame()  # TODO remove

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


'''
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

'''

'''def plot_3d_figure(point_cloud, normals, colors_np):
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
'''

# depth, rgb, c2w = read_data(data, 0)
# H, W = depth.shape

# vertices = ICP.compute_vertices(depth, K)
# normals = ICP.compute_normals(vertices)
# colors = np.array([1, 0, 0]).reshape(1, 3).repeat(H*W, axis=0)

# R = c2w[:3, :3]
# t = c2w[:3, -1]
# vertices = torch.matmul(vertices, R.T) + t
# pcd = plot_3d_figure(vertices, normals, colors)


# c2w_list += [c2w.cpu().numpy()]
# c2w_gt_list += [c2w1.cpu().numpy()]

'''
avg_time = np.array(time_list).mean()
print("average processing time: {:f}s per frame, i.e. {:f} fps".format(avg_time, 1. / avg_time))

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


vertices1 = ICP.compute_vertices(depth1, K)
normals1 = ICP.compute_normals(vertices1)
colors1 = np.array([0, 0, 1]).reshape(1, 3).repeat(H*W, axis=0)

R = c2w[:3, :3]
t = c2w[:3, -1]
vertices1 = torch.matmul(vertices1, R.T) + t
pcd1 = plot_3d_figure(vertices1, normals1, colors1)

o3d.visualization.draw_geometries([pcd, pcd1], point_show_normal=False)
'''
