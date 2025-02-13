from primesense import openni2
from PyQt5.QtCore import *
import torch
import numpy as np

from icp.levenberg_marquardt import LM_optimizer
from icp.icp import ICP
import volume_ray_final as tsdf
from bilateral_filter import bilateral_filtering
from block_averaging_subsampling import block_averaging


class SensorWorker(QThread):
    new_frame = pyqtSignal(object)
    new_grid = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        # self.is_running = threading.Event()
        self.is_running = True
        self.cuda_device = None
        self.refresh_rate_mesh_render = 5
        self.counter = self.refresh_rate_mesh_render - 1
        self.num_scales = 3

        if torch.cuda.is_available():
            print("Using CUDA")
            self.cuda_device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            print("Using MPS")
            self.cuda_device = torch.device("mps")
        else:
            print("Using CPU")
            self.cuda_device = torch.device("cpu")

        self.optimizer = LM_optimizer(max_iterations=10)
        self.icp = ICP(optimizer=None, occlusion_threshold=0.1,
                       symmetric_error=True)

        self.icp_solvers = [
            ICP(optimizer=LM_optimizer(max_iterations=6, damping_factor=1.0e-4),
                occlusion_threshold=0.1, symmetric_error=True),
            ICP(optimizer=LM_optimizer(max_iterations=3, damping_factor=1.0e-4),
                occlusion_threshold=0.1, symmetric_error=True),
            ICP(optimizer=LM_optimizer(max_iterations=3, damping_factor=1.0e-2),
                occlusion_threshold=0.1, symmetric_error=True)
        ]

        self.multiscales = [torch.nn.MaxPool2d(
            1 << i, 1 << i) for i in range(self.num_scales)]

        # Constants
        # TODO cannot halve those on kinect
        self.height = 480 // 2
        self.width = 640 // 2
        self.height_l2 = self.height // 2
        self.width_l2 = self.width // 2
        self.height_l3 = self.height_l2 // 2
        self.width_l3 = self.width_l2 // 2
        fps = 30
        dist = "/home/zeus/Install/kinect/openni2/OpenNI2/Packaging/OpenNI2-x64/Redist/"
        # can also accept the path of the OpenNI redistribution
        openni2.initialize(dist)

        self.dev = openni2.Device.open_any()
        self.dev.set_depth_color_sync_enabled(True)

        self.depth_stream = self.dev.create_depth_stream()
        self.color_stream = self.dev.create_color_stream()

        self.depth_stream.configure_mode(
            self.width, self.height, fps, openni2.PIXEL_FORMAT_DEPTH_1_MM)
        self.color_stream.configure_mode(
            self.width, self.height, fps, openni2.PIXEL_FORMAT_RGB888)
        self.depth_stream.start()
        self.color_stream.start()

        self.dev.set_image_registration_mode(
            openni2.IMAGE_REGISTRATION_DEPTH_TO_COLOR)

        self.depth_max = self.depth_stream.get_max_pixel_value()
        self.depth_min = 1
        # TODO another one of those Kinect crashes
        # self.depth_min = self.depth_stream.get_min_pixel_value()

        fx = (0.5 * self.width) // np.tan(0.5 *
                                          self.depth_stream.get_horizontal_fov())
        fy = (0.5 * self.height) // np.tan(0.5 *
                                           self.depth_stream.get_vertical_fov())
        px = self.width // 2.0
        py = self.height // 2.0

        self.k = torch.tensor([[fx, 0, px], [0, fy, py], [0, 0, 1]]).to(
            dtype=torch.float32).to(self.cuda_device)

        self.c2w = torch.eye(4, dtype=torch.float32, device=self.cuda_device)
        self.c2w[0, 3] = -0.25  # -0.25
        self.c2w[1, 3] = 1.0  # 1.0
        self.c2w[2, 3] = -0.1
        self.volume_bounds = None
        self.vox_grid = None
        self.last_frame = None
        self.current_frame = None

        self.sigma_spatial = 30
        self.sigma_range = 250

    def run(self):
        while self.is_running:
            self.process_frame()
            self.new_frame.emit(self.current_frame)
            if self.counter == 0:  # only update mesh every tenth frame?
                self.new_grid.emit(self.vox_grid)
                # self.counter = 0
            self.counter = (self.counter + 1) % self.refresh_rate_mesh_render
            print(self.counter)

    def stop(self):
        self.is_running = False
        self.wait()
        self.depth_stream.stop()
        self.color_stream.stop()
        openni2.unload()

    def reset(self):
        # TODO: more logic for resetting reconstruction?
        self.last_frame = None
        self.c2w = torch.eye(4, dtype=torch.float64, device=self.cuda_device)
        self.c2w[0, 3] = -0.25  # -0.25
        self.c2w[1, 3] = 1.0  # 1.0
        self.c2w[2, 3] = -0.1
        self.counter = 0
        print("reset reconstruction")

    def read_frame(self):
        depth_frame = self.depth_stream.read_frame()
        color_frame = self.color_stream.read_frame()

        depth_frame_data = np.frombuffer(
            depth_frame.get_buffer_as_uint16(), dtype=np.uint16)
        depth_frame_data = torch.from_numpy(depth_frame_data.astype(np.float32)).to(
            self.cuda_device).reshape((self.height, self.width))
        # depth_frame_data = torch.flip(depth_frame_data, dims=[1, 0])

        color_frame_data = np.frombuffer(
            color_frame.get_buffer_as_uint8(), dtype=np.uint8)
        color_frame_data = torch.from_numpy(color_frame_data).to(
            self.cuda_device).reshape(self.height, self.width, 3)
        # color_frame_data = torch.flip(color_frame_data, dims=[1, 0])

        depth_frame_data /= 1000.0
        depth_frame_data[(depth_frame_data < 0.1) |
                         (depth_frame_data > 5.0)] = 0.0

        return color_frame_data, depth_frame_data

    def process_frame(self):
        color_map, depth_map = self.read_frame()
        h, w = depth_map.shape
        self.current_frame = [color_map, depth_map]

        if self.last_frame is None:
            volume_bounds = tsdf.get_vol_bnds(
                depth_map, self.k.cpu().numpy(), self.c2w.cpu().numpy())
            self.vox_grid = tsdf.TSDF(vol_dim=volume_bounds, intristics=self.k)
            self.last_frame = self.current_frame
            self.vox_grid.integrate(depth_map, self.c2w, color_map)
            return

        # R = self.c2w[:3, :3]
        # t = self.c2w[:3, -1]

        depth1, mask1 = self.vox_grid.render_model(self.c2w, self.k, h, w,
                                                   near=0.25, far=5., n_samples=192)

        dpt_curr_pyr = [f(depth_map.view(1, 1, h, w))
                        for f in self.multiscales]
        dpt_curr_pyr = [d.squeeze() for d in dpt_curr_pyr]
        dpt1_pyr = [f(depth1.view(1, 1, h, w)) for f in self.multiscales]
        dpt1_pyr = [d.squeeze() for d in dpt1_pyr]

        T10 = torch.eye(4, dtype=torch.float32, ).to(self.cuda_device)

        try:
            for j in reversed(range(self.num_scales)):
                K_scaled = self.k.clone()
                if j != 0:
                    K_scaled[0, 0] /= 2 ** j
                    K_scaled[1, 1] /= 2 ** j
                    K_scaled[0, 2] /= 2 ** j
                    K_scaled[1, 2] /= 2 ** j

                T10, err_msg = self.icp_solvers[j](
                    dpt_curr_pyr[j], dpt1_pyr[j], T10, K_scaled)
                if err_msg:
                    print("ERROR:", err_msg)
                else:
                    print("No error")
        except Exception as e:
            print(e)
        else:
            self.c2w = self.c2w @ T10
            self.vox_grid.integrate(depth_map, self.c2w, color_map)
