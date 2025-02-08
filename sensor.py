from primesense import openni2
from PyQt5.QtCore import *

from icp.levenberg_marquardt import LM_optimizer
from icp.icp import ICP
from volume_ray_final import *
from bilateral_filter import bilateral_filtering
from block_averaging_subsampling import block_averaging


class SensorWorker(QThread):
    new_frame = pyqtSignal(object)
    new_grid = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        #self.is_running = threading.Event()
        self.is_running = True
        self.cuda_device = None
        self.counter = 1

        if torch.cuda.is_available():
            print("Using CUDA")
            self.cuda_device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            print("Using MPS")
            self.cuda_device = torch.device("mps")
        else:
            print("Using CPU")
            self.cuda_device = torch.device("cpu")

        self.optimizer1 = LM_optimizer(max_iterations=6)
        self.optimizer2 = LM_optimizer(max_iterations=3)
        self.optimizer3 = LM_optimizer(max_iterations=3)
        self.icp1 = ICP(optimizer=self.optimizer1, symmetric_error=True)
        self.icp2 = ICP(optimizer=self.optimizer2, symmetric_error=True)
        self.icp3 = ICP(optimizer=self.optimizer3, symmetric_error=True)


        # Constants
        self.height = 480
        self.width = 640
        self.height_l2 = self.height // 2
        self.width_l2 = self.width // 2
        self.height_l3 = self.height_l2 // 2
        self.width_l3 = self.width_l2 // 2
        fps = 30

        openni2.initialize()  # can also accept the path of the OpenNI redistribution

        self.dev = openni2.Device.open_any()
        self.dev.set_depth_color_sync_enabled(True)

        self.depth_stream = self.dev.create_depth_stream()
        self.color_stream = self.dev.create_color_stream()

        self.depth_stream.configure_mode(self.width, self.height, fps, openni2.PIXEL_FORMAT_DEPTH_1_MM)
        # self.color_stream.configure_mode(self.width, height, fps, openni2.PIXEL_FORMAT_RGB888)

        self.depth_stream.start()
        self.color_stream.start()

        self.dev.set_image_registration_mode(openni2.IMAGE_REGISTRATION_DEPTH_TO_COLOR)

        cam_settings = None  # color_stream.camera
        print(cam_settings)

        if cam_settings is not None:
            print("Disabling auto exposure and white balance")
            cam_settings.set_auto_exposure(False)
            cam_settings.set_auto_white_balance(False)
            self.color_stream.camera = cam_settings
        else:
            print("No cam settings")

        self.depth_max = self.depth_stream.get_max_pixel_value()
        self.depth_min = 1 # self.depth_stream.get_min_pixel_value()

        fx = (0.5 * self.width) // np.tan(0.5 * self.depth_stream.get_horizontal_fov())
        fy = (0.5 * self.height) // np.tan(0.5 * self.depth_stream.get_vertical_fov())
        px = self.width // 2.0
        py = self.height // 2.0

        self.k1 = np.array([
            [fx, 0.0, px],
            [0.0, fy, py],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)
        self.k2 = np.array([
            [0.5 * fx, 0.0, 0.5 * px],
            [0.0, 0.5 * fy, 0.5 * py],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)
        self.k3 = np.array([
            [0.25 * fx, 0.0, 0.25 * px],
            [0.0, 0.25 * fy, 0.25 * py],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)
        self.k_pyr = [self.k1, self.k2, self.k3]

        self.c2w = np.eye(4)
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
                #self.counter = 0
            self.counter = (self.counter + 1) % 10

    def stop(self):
        self.is_running = False
        self.wait()
        self.depth_stream.stop()
        self.color_stream.stop()
        openni2.unload()

    def calc_icp(self, depth_pyr, tsdf_depth_pyr):
        t10 = self.icp3(depth_pyr[2],
                   tsdf_depth_pyr[2],
                   torch.tensor(np.identity(4), dtype=torch.float32).to(self.cuda_device),
                   torch.tensor(self.k_pyr[2], dtype=torch.float32).to(self.cuda_device))
        # print("icp l3")
        # print(t10)
        t10 = self.icp2(depth_pyr[1],
                   tsdf_depth_pyr[1],
                   t10,
                   torch.tensor(self.k_pyr[1], dtype=torch.float32).to(self.cuda_device))
        # print("icp l2")
        # print(t10)
        t10 = self.icp1(depth_pyr[0],
                   tsdf_depth_pyr[0],
                   t10,
                   torch.tensor(self.k_pyr[0], dtype=torch.float32).to(self.cuda_device))
        # print("icp l1")
        # print(t10)
        # TODO range/instead check for Null?
        '''    if torch.allclose(t10, torch.eye(4).to(cuda_device), atol=1e-3):  #TODO
            print("ICP failed or did not improve pose")
        else:
            c2w = c2w @ t10.cpu().numpy()'''
        self.c2w = self.c2w @ t10.cpu().numpy()

    def create_depth_pyramid(self, depth_frame_data):
        depth_map_l1, validity_mask = bilateral_filtering(
            depth_image=depth_frame_data,
            kernel_size=21,
            sigma_spatial=self.sigma_spatial,
            sigma_range=self.sigma_range,
            min_depth=self.depth_min,
            max_depth=self.depth_max
        )
        depth_map_l2 = block_averaging(
            depth_map_l1, 2, self.sigma_range
        )
        depth_map_l3 = block_averaging(
            depth_map_l2, 2, self.sigma_range
        )

        dep_pyr = [depth_map_l1, depth_map_l2, depth_map_l3]

        # TODO maybe adjust and maybe move
        dep_pyr = [d / 1000.0 for d in dep_pyr]

        return dep_pyr

    def reset(self):
        # TODO: more logic for resetting reconstruction?
        self.last_frame = None
        print("reset reconstruction")

    def read_frame(self):
        depth_frame = self.depth_stream.read_frame()
        color_frame = self.color_stream.read_frame()

        depth_frame_data = torch.frombuffer(depth_frame.get_buffer_as_uint16(), dtype=torch.uint16).reshape(
            (self.height, self.width))
        depth_frame_data = depth_frame_data.to(torch.float32).to(self.cuda_device)
        color_frame_data = torch.frombuffer(color_frame.get_buffer_as_uint8(), dtype=torch.uint8).reshape(
            (self.height, self.width, 3))

        # TODO ?
        depth_frame_data[depth_frame_data == 65535] = 0
        return color_frame_data, depth_frame_data

    def process_frame(self):
        color_map, depth_map = self.read_frame()
        dep_pyr = self.create_depth_pyramid(depth_map)

        self.current_frame = [color_map, dep_pyr]

        if self.last_frame is None:
            volume_bounds = get_vol_bnds(dep_pyr[0], self.k_pyr[0], self.c2w)
            self.vox_grid = TSDF(vol_dim=volume_bounds, intristics=self.k_pyr[0])
            self.last_frame = self.current_frame
            self.vox_grid.integrate(self.last_frame[1][0], self.c2w, self.last_frame[0])
            print(self.vox_grid)
            return

        # TODO change 2nd param
        self.calc_icp(dep_pyr, self.last_frame[1])
        #self.vox_grid.integrate(self.current_frame[1][0], self.c2w, self.current_frame[0])

