import numpy as np
import torch
import MeasurementModule
import time
from icp.levenberg_marquardt import LM_optimizer
from icp.icp import ICP
import threading

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

processing_running = True
optimizer = LM_optimizer(max_iterations=5)
icp = ICP(optimizer=None, occlusion_threshold=1, symmetric_error=True)


def init_worker():
    MeasurementModule.Init()


def processing_worker():
    MeasurementModule.StartProcessingThread()


def frame_acquisition_worker() :
    while not MeasurementModule.FrameCallback(pythonCallback):
        print("trying\n")
        time.sleep(1)
    print("Queue Initialized!\n")


# TODO rename this func
def pythonCallback():
    print("2 frames available.")
    process_frames()


def process_frames():
    current_frame = None
    last_frame = None
    d_max = 10000  # define allowed range of depth values in mm
    d_min = 0  # define allowed range of depth values in mm
    K_tensor_l1 = None
    K_tensor_l2 = None
    K_tensor_l3 = None
    c2w = None

    print("Processing frames...")

    while processing_running:
        # Init Logic for the first frame
        if last_frame is None:
            print("Initializing first frame...")
            last_frame = MeasurementModule.PopFrame()
            c2w = torch.tensor(np.eye(4), dtype=torch.float32).to(device)
            K_tensor_l1 = torch.tensor(MeasurementModule.Device.K()).to(device)
            K_tensor_l2 = torch.tensor(MeasurementModule.Device.K2()).to(device)
            K_tensor_l3 = torch.tensor(MeasurementModule.Device.K3()).to(device)
            continue

        # Process subsequent frames
        start_time = time.time()

        current_frame = MeasurementModule.PopFrame()

        print("Calculating ICP...")
        # ICP Processing
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

        c2w = c2w @ T10
        # print("C2W!", c2w)
        print("ICP...Done!")

        # Record end time and write duration to a file
        end_time = time.time()
        elapsed_time = end_time - start_time

        with open("C:/Users/steph/Documents/Projekte/KinectFusion/processing_time.txt", "a") as f:
            f.write(f"{elapsed_time}\n")


frame_acquisition_thread = threading.Thread(target=frame_acquisition_worker)
init_thread = threading.Thread(target=init_worker)
processing_thread = threading.Thread(target=processing_worker)

# Main Program
init_worker()
time.sleep(2)
print("yolo")
frame_acquisition_thread.start()
processing_thread.start()
time.sleep(100)
print("bolo")
#


