import sys
import numpy as np
from skimage import measure
from scipy.interpolate import RegularGridInterpolator

import OpenGL.GL as gl
from OpenGL import GLU
from OpenGL.arrays import vbo

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from sensor import SensorWorker


class MeshWidget(QOpenGLWidget):
    def __init__(self, parent=None):
        super(MeshWidget, self).__init__(parent)
        self.vox_grid = None
        self.setFixedSize(600, 600)

        self.rotX = 0.0
        self.rotY = 0.0
        self.rotZ = 0.0

        self.verts = []
        self.faces = []
        self.vertex_colors = []
        self.normals = []

        self.color_mode = False
        self.vertVBO = None
        self.colorVBO = None
        self.normalVBO = None

    def closeEvent(self, event):
        self.makeCurrent()
        self.doneCurrent()
        self.deleteLater()
        event.accept()

    def initializeGL(self):
        gl.glClearColor(0.2, 0.2, 0.2, 1.0)

        self.vertVBO = vbo.VBO(np.array([], dtype=np.float32))
        self.colorVBO = vbo.VBO(np.array([], dtype=np.float32))
        self.normalVBO = vbo.VBO(np.array([], dtype=np.float32))

    def resizeGL(self, width, height):
        gl.glViewport(0, 0, width, height)
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        aspect = width / float(height)

        GLU.gluPerspective(45.0, aspect, 0.1, 10.0)
        gl.glMatrixMode(gl.GL_MODELVIEW)


    def paintGL(self):
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_LIGHTING)
        gl.glEnable(gl.GL_LIGHT0)
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_POSITION, [2., 2., -10., 0.])
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_DIFFUSE, [1.0, 1.0, 1.0])
        gl.glLightModelfv(gl.GL_LIGHT_MODEL_AMBIENT, [0.2, 0.2, 0.2, 1.0])

        gl.glShadeModel(gl.GL_SMOOTH)

        gl.glPushMatrix()

        gl.glTranslate(0.0, 0.0, -5.0)
        gl.glScale(3.0, 3.0, 3.0)
        gl.glRotate(self.rotX+180, 1.0, 0.0, 0.0)
        gl.glRotate(self.rotY, 0.0, 1.0, 0.0)
        gl.glRotate(self.rotZ, 0.0, 0.0, 1.0)

        gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
        gl.glEnableClientState(gl.GL_COLOR_ARRAY)
        gl.glEnableClientState(gl.GL_NORMAL_ARRAY)

        self.vertVBO.bind()
        gl.glVertexPointer(3, gl.GL_FLOAT, 0, self.vertVBO)
        self.colorVBO.bind()
        gl.glColorPointer(3, gl.GL_FLOAT, 0, self.colorVBO)
        self.normalVBO.bind()
        gl.glNormalPointer(gl.GL_FLOAT, 0, self.normalVBO)

        gl.glDrawElements(gl.GL_TRIANGLES, len(self.faces)*3, gl.GL_UNSIGNED_INT, self.faces)

        gl.glDisableClientState(gl.GL_VERTEX_ARRAY)
        gl.glDisableClientState(gl.GL_COLOR_ARRAY)

        gl.glPopMatrix()

    def write_mesh_file(self):
        filename = "reconstruction.off"
        with open(filename, 'w') as f:
            if self.color_mode:
                f.write("COFF\n")
            else:
                f.write("OFF\n")
            f.write(f"{len(self.verts)} {len(self.faces)} 0\n")

            for i, v in enumerate(self.verts):
                vertex_str = f"{v[0]} {v[1]} {v[2]}"
                if self.color_mode:
                    color_int = (self.vertex_colors[i] * 255).astype(int)
                    color_str = ' '.join(map(str, color_int))
                    f.write(f"{vertex_str} {color_str}\n")
                else:
                    f.write(f"{vertex_str}\n")

            for face in self.faces:
                f.write(f"{len(face)} {' '.join(map(str, face))}\n")

    def get_mesh(self):
        if self.vox_grid is not None:
            sdf_numpy = self.vox_grid.sdf_values.cpu().numpy()
            color_sdf = self.vox_grid.rgb_values.cpu().numpy()
            voxel_size = 0.02
            # voxel_size = 20

            verts, faces, norms, vals = measure.marching_cubes(sdf_numpy, level=0)
            verts_ind = np.round(verts).astype(int)
            self.verts = verts * voxel_size
            self.faces = faces

            # TODO this crashes regularly
            #y = np.arange(color_sdf.shape[1]) * voxel_size
            #x = np.arange(color_sdf.shape[0]) * voxel_size
            #z = np.arange(color_sdf.shape[2]) * voxel_size

            #r_interpolator = RegularGridInterpolator((x, y, z), color_sdf[..., 0])
            #g_interpolator = RegularGridInterpolator((x, y, z), color_sdf[..., 1])
            #b_interpolator = RegularGridInterpolator((x, y, z), color_sdf[..., 2])

            #r_values = r_interpolator(verts)
            #g_values = g_interpolator(verts)
            #b_values = b_interpolator(verts)

            #self.vertex_colors = np.stack((b_values, g_values, r_values), axis=-1)
            default_color = np.array([0.5, 0.5, 1.0])  # RGB color for blue

            # Set all vertex colors to the default color
            self.vertex_colors = np.tile(default_color, (self.verts.shape[0], 1))

            min_bound = np.min(self.verts, axis=0)
            max_bound = np.max(self.verts, axis=0)
            center = (min_bound + max_bound) / 2.0
            self.verts -= center  # Shift mesh to origin

            # Scale mesh to fit within [-1, 1] range
            max_extent = np.max(max_bound - min_bound)
            self.verts /= max_extent  # Normalize size

            self.vertVBO.set_array(np.reshape(self.verts,(1, -1)).astype(np.float32))
            self.colorVBO.set_array(np.reshape(self.vertex_colors,(1, -1)).astype(np.float32))
            self.normalVBO.set_array(np.reshape(norms,(1, -1)).astype(np.float32))


    def update_grid(self, new_vox_grid):
        # Update voxel grid
        self.vox_grid = new_vox_grid
        self.get_mesh()
        self.update()

    def set_rot_x(self, value):
        self.rotX = value
        self.update()

    def set_rot_y(self, value):
        self.rotY = value
        self.update()

    def set_rot_z(self, value):
        self.rotZ = value
        self.update()

    def reset(self):
        self.vox_grid = None


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.sensor = SensorWorker()
        self.mesh_panel = MeshWidget()
        self.color_panel = QLabel()
        self.depth_panel = QLabel()

        self.sliders = []
        self.setup_gui()

        self.sensor.new_frame.connect(self.get_img_from_frame)
        self.sensor.new_grid.connect(self.mesh_panel.update_grid)

        self.sensor.start()

    def setup_gui(self):
        self.setWindowTitle("Kinect Fusion")
        central_widget = QWidget()
        xSlider = QSlider(Qt.Horizontal)
        ySlider = QSlider(Qt.Horizontal)
        zSlider = QSlider(Qt.Horizontal)
        xSlider.setRange(-360, 360)
        xSlider.setValue(0)
        ySlider.setRange(-360, 360)
        ySlider.setValue(0)
        zSlider.setRange(-360, 360)
        zSlider.setValue(0)

        self.sliders = [xSlider, ySlider, zSlider]

        reset_btn = QPushButton("Reset")
        save_btn = QPushButton("Save Mesh")

        buttonLayout = QHBoxLayout()
        buttonLayout.addWidget(xSlider)
        buttonLayout.addWidget(ySlider)
        buttonLayout.addWidget(zSlider)
        buttonLayout.addWidget(reset_btn)
        buttonLayout.addWidget(save_btn)

        input_layout = QVBoxLayout()
        input_layout.addWidget(self.color_panel)
        input_layout.addWidget(self.depth_panel)
        input_layout.addLayout(buttonLayout)

        main_layout = QHBoxLayout()
        main_layout.addWidget(self.mesh_panel)
        main_layout.addLayout(input_layout)

        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        reset_btn.clicked.connect(self.reset)
        save_btn.clicked.connect(self.mesh_panel.write_mesh_file)
        xSlider.valueChanged.connect(self.mesh_panel.set_rot_x)
        ySlider.valueChanged.connect(self.mesh_panel.set_rot_y)
        zSlider.valueChanged.connect(self.mesh_panel.set_rot_z)

    def closeEvent(self, event):
        if self.sensor.is_running:
            print("stopping sensor thread")
            self.sensor.stop()
            self.sensor.wait()
            self.sensor.deleteLater()
            print("stopped thread")
        self.close()

    def reset(self):
        self.sensor.reset()
        self.mesh_panel.reset()

    def get_img_from_frame(self, frame):
        color_map = frame[0].cpu().numpy()
        depth_map = frame[1][0].cpu().numpy()

        h, w, ch = color_map.shape
        bytes_per_line = ch * w
        color_img = QImage(color_map.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)

        depth_min, depth_max = depth_map.min(), depth_map.max()
        if depth_max > depth_min:  # Avoid division by zero
            depth_normalized = ((depth_map - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
        else:
            depth_normalized = np.zeros_like(depth_map, dtype=np.uint8)  # If no valid depth values
        depth_img = QImage(depth_normalized.data, w, h, w, QImage.Format.Format_Grayscale8)

        self.color_panel.setPixmap(QPixmap.fromImage(color_img).scaledToHeight(240))
        self.depth_panel.setPixmap(QPixmap.fromImage(depth_img).scaledToHeight(240))