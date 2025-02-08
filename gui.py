import sys
import numpy as np

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from sensor import SensorWorker

class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setWindowTitle("Kinect Fusion")
        self.sensor = SensorWorker()
        central_widget = QWidget()
        self.mesh_panel = QLabel()
        self.color_panel = QLabel()
        self.depth_panel = QLabel()
        input_layout = QVBoxLayout()
        input_layout.addWidget(self.color_panel)
        input_layout.addWidget(self.depth_panel)

        main_layout = QHBoxLayout()
        main_layout.addWidget(self.mesh_panel)
        main_layout.addLayout(input_layout)

        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        self.sensor.new_frame.connect(self.get_img_from_frame)

        self.sensor.start()

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


if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
