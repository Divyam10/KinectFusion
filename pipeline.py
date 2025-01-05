import os

DATA_DIR = 'data/rgbd_dataset_freiburg1_desk'
DATA_PATH = os.path.join(os.getcwd(), DATA_DIR)

depth_file = os.path.join(DATA_PATH, 'depth.txt')
rgb_file = os.path.join(DATA_PATH, 'rgb.txt')
trajectory_file = os.path.join(DATA_PATH, 'groundtruth.txt')




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
print(len(data), data[:20])´



