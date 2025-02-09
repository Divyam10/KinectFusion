import torch


#TODO this is just copy pasted from icp branch/pipeline.py
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
