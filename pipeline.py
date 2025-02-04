import sys
import numpy as np
import torch
import MeasurementModule
import time
from icp.levenberg_marquardt import LM_optimizer
from icp.icp import ICP
from volume_ray_final import *
import threading

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

processing_running = True
frame_acquisition_thread = None
init_thread = None
processing_thread = None
optimizer = LM_optimizer(max_iterations=5)
icp = ICP(optimizer=optimizer, occlusion_threshold=1, symmetric_error=True)


def init_worker():
    MeasurementModule.Init()


def processing_worker():
    MeasurementModule.StartProcessingThread()


def frame_acquisition_worker():
    while not MeasurementModule.FrameCallback(pythonCallback):
        print("trying\n")
        time.sleep(1)
    print("Queue Initialized!\n")


# TODO rename this func
def pythonCallback():
    print("2 frames available.")
    process_frames()


def process_frames():
    last_frame = None
    d_max = 10000  # define allowed range of depth values in mm
    d_min = 0  # define allowed range of depth values in mm
    k_l_1 = None
    k_l_2 = None
    k_l_3 = None
    c2w = np.eye(4)
    vox_grid = None

    print("Processing frames...")

    while processing_running:
        # Init Logic for the first frame
        if last_frame is None:
            print("Initializing first frame...")
            last_frame = MeasurementModule.PopFrame()

            k_l_1 = MeasurementModule.Device.K()
            k_l_2 = MeasurementModule.Device.K2()
            k_l_3 = MeasurementModule.Device.K3()

            print("Computing volume bounds...")
            volume_bounds = get_vol_bnds(last_frame.l1.depth_map, k_l_1, c2w)
            print("Computing voxel grid...")
            vox_grid = TSDF(volume_bounds, voxel_size=0.02, intristics=k_l_1)
            print("Voxel grid... Done!")
            continue

        # On new frame:
        current_frame = MeasurementModule.PopFrame()

        if current_frame is None:
            print("Waiting for Frame...")
            time.sleep(0.03)
            continue

        depth_frame = current_frame.l1.depth_map

        print("Preprocessing frame...")
        depth_frame[depth_frame == 65535] = 0

        # TODO check near/far
        print("Synthesize model depth frame")
        dep_pyr, rgb_pyr, vtx_pyr, nrm_pyr, mask_pyr = vox_grid.render_pyramid(
            c2w=c2w,
            intri=k_l_1,
            imh=240,
            imw=320,
            n_pyr=3,
            near=0.5,
            far=5.0
        )
        print("Synthesis... Done!")

        print("Calculating ICP...")
        c2w = calc_icp(current_frame, dep_pyr, k_l_1, k_l_2, k_l_3, c2w)
        print("ICP... Done!")

        print("Integrate new depth and color into model")
        vox_grid.integrate(depth_frame, c2w, current_frame.color_map)
        print("Integration... Done!")

        print("Performing Marching Cubes...")
        get_mesh(vox_grid)
        print("Mesh generation... Done!")
        continue


def calc_icp(
        current_frame: MeasurementModule.ProcessedFrame,
        tsdf_depth_pyramid: list,
        k_l_1: np.ndarray,
        k_l_2: np.ndarray,
        k_l_3: np.ndarray,
        c2w: np.ndarray
):
    t10 = icp(torch.tensor(current_frame.l3.depth_map, dtype=torch.float32).to(device),
              torch.tensor(tsdf_depth_pyramid[2], dtype=torch.float32).to(device),
              torch.tensor(np.identity(4), dtype=torch.float32).to(device),
              torch.tensor(k_l_3, dtype=torch.float32).to(device))

    t10 = icp(torch.tensor(current_frame.l2.depth_map, dtype=torch.float32).to(device),
              torch.tensor(tsdf_depth_pyramid[1], dtype=torch.float32).to(device),
              t10,
              torch.tensor(k_l_2, dtype=torch.float32).to(device))

    t10 = icp(torch.tensor(current_frame.l1.depth_map, dtype=torch.float32).to(device),
              torch.tensor(tsdf_depth_pyramid[0], dtype=torch.float32).to(device),
              t10,
              torch.tensor(k_l_1, dtype=torch.float32).to(device))

    # TODO range/instead check for Null?
    '''if torch.allclose(t10, torch.eye(4).to(device), atol=1e-3):
        print("ICP failed or did not improve pose")
    else:
        c2w = c2w @ t10'''

    c2w = c2w @ t10
    return c2w


def main() -> int:
    global frame_acquisition_thread, init_thread, processing_thread, processing_running

    frame_acquisition_thread = threading.Thread(target=frame_acquisition_worker)
    init_thread = threading.Thread(target=init_worker)
    processing_thread = threading.Thread(target=processing_worker)
    user_input = None

    init_worker()
    frame_acquisition_thread.start()
    processing_thread.start()

    while user_input is None:
        user_input = input("Press Something to exit: ")
        time.sleep(1)

    processing_running = False
    MeasurementModule.Device.set_cxx_running(False)
    frame_acquisition_thread.join()
    processing_thread.join()

    print("All threads stopped. Exiting.")
    return 0


if __name__ == '__main__':
    sys.exit(main())

# start_time = time.time()
# end_time = time.time()
# elapsed_time = end_time - start_time
# with open("C:/Users/steph/Documents/Projekte/KinectFusion/processing_time.txt", "a") as f:
# f.write(f"{elapsed_time}\n")
