import threading
import numpy as np
import torch
import MeasurementModule
import time
from icp.levenberg_marquardt import LM_optimizer
from icp.icp import ICP

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# Event flags for synchronization
processing_running = threading.Event()
frame_available = threading.Event()

def on_new_frame():
    global frame_available
    # Signal C++ to not call frame callbacks while Python processes
    MeasurementModule.Device.set_python_processing(True)
    print("New frame available.")
    frame_available.set()  # Notify processing thread to start

def process_frames():
    global frame_available
    current_frame = None
    last_frame = None
    d_max = 10000  # define allowed range of depth values in mm
    d_min = 0  # define allowed range of depth values in mm
    K_tensor_l1 = None
    K_tensor_l2 = None
    K_tensor_l3 = None
    c2w = None

    print("Processing frames...")

    while processing_running.is_set():
        print("Waiting for frames...")
        # Wait for a frame to be available (this blocks until the event is set)
        frame_available.wait()
        print("Got frame...")

        # Init Logic for the first frame
        if last_frame is None:
            print("Initializing first frame...")
            last_frame = MeasurementModule.PopFrame()
            c2w = torch.tensor(np.eye(4), dtype=torch.float32).to(device)
            K_tensor_l1 = torch.tensor(MeasurementModule.Device.K()).to(device)
            K_tensor_l2 = torch.tensor(MeasurementModule.Device.K2()).to(device)
            K_tensor_l3 = torch.tensor(MeasurementModule.Device.K3()).to(device)
            # Skip to next iteration after initialization is done
            frame_available.clear()  # Clear event, wait for next frame
            # Signal C++ that it can/should call callbacks
            MeasurementModule.Device.set_python_processing(False)
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
        #print("C2W!", c2w)
        print("ICP...Done!")

        # Record end time and write duration to a file
        end_time = time.time()
        elapsed_time = end_time - start_time

        with open("C:/Users/steph/Documents/Projekte/KinectFusion/processing_time.txt", "a") as f:
            f.write(f"{elapsed_time}\n")

        frame_available.clear()  # Wait for next frame

        # Signal C++ that it can/should call callbacks
        MeasurementModule.Device.set_python_processing(False)
        # time.sleep(0.01)

def start_cpp_init():
    # This method will start the C++ initialization in a background thread
    MeasurementModule.Init(on_new_frame)

# Initialize ICP optimizer
optimizer = LM_optimizer(max_iterations=5)
icp = ICP(optimizer=None, occlusion_threshold=1, symmetric_error=True)

# Start processing thread
processing_running.set()  # Signal to start processing
processing_thread = threading.Thread(target=process_frames, daemon=True)
processing_thread.start()

# Start the C++ initialization in a worker thread
cpp_init_thread = threading.Thread(target=start_cpp_init, daemon=True)
cpp_init_thread.start()

# Keep the main thread alive while other threads do their work
cpp_init_thread.join()  # Wait for C++ initialization to complete (optional)
