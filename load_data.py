import numpy as np
import cv2
import pathlib
import torch
import os
import glob


def load_rgb(path_to_folder):

    images = glob.glob(path_to_folder + "/" + "*.color.jpg")
    for image in images:
        print(image)

    return 0


def load_pose(path_to_folder):
    poses = glob.glob(path_to_folder + "/" + "*.pose.txt")
    for pose in poses:
        print(pose)

    return 0


def load_depth(path_to_folder):
    depths = glob.glob(path_to_folder + "/" + "*.depth.png")
    for depth in depths:
        print(depth)
    return 0


if __name__ == "__main__":
    load_rgb("/home/zeus/masters/3DSMC/project/TSDF/data")
    load_depth("/home/zeus/masters/3DSMC/project/TSDF/data")
    load_pose("/home/zeus/masters/3DSMC/project/TSDF/data")
    print("Hello")
