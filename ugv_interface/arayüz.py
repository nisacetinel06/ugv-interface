import sys
import socket
import struct
import threading
import pickle
import json
import asyncio
import cv2
import websockets
import numpy as np
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QPoint
from PyQt5.QtGui import QImage, QPixmap, QPainter
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout,
    QHBoxLayout, QGridLayout, QFrame
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import os

# --- Radar Paneli ---
class RadarWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(300, 200)
        self.angle = 90

    def mouseMoveEvent(self, event):
        x = event.x()
        width = self.width()
        self.angle = max(0, min(180, int((x / width) * 180)))
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), Qt.black)

        pen = painter.pen()
        pen.setColor(Qt.green)
        pen.setWidth(1)
        painter.setPen(pen)

        center = self.rect().bottomLeft() + QPoint(self.width() // 2, 0)
        radius = min(self.width(), self.height()) - 10

        for r in range(1, 4):
            painter.drawArc(center.x() - r * 30, center.y() - r * 30, r * 60, r * 60, 0 * 16, 180 * 16)

        for angle in range(0, 181, 30):
            rad = np.radians(angle)
            x = center.x() + radius * np.cos(rad)
            y = center.y() - radius * np.sin(rad)
            painter.drawLine(center.x(), center.y(), int(x), int(y))

        radar_pen = painter.pen()
        radar_pen.setColor(Qt.green)
        radar_pen.setWidth(2)
        painter.setPen(radar_pen)

        rad = np.radians(self.angle)
        x = center.x() + radius * np.cos(rad)
        y = center.y() - radius * np.sin(rad)
        painter.drawLine(center.x(), center.y(), int(x), int(y))
        painter.drawText(10, 20, f"Açı: {self.angle}°")
# --- Jiroskop 3D Gösterimi ---
class GyroCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(2.5, 2.5))
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(111, projection="3d")
        self._init_plot()

    def _init_plot(self):
        self.ax.set_xlim([-1, 1])
        self.ax.set_ylim([-1, 1])
        self.ax.set_zlim([-1, 1])
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")

    def draw_arrow(self, alpha, beta, gamma):
        self.ax.cla()
        self._init_plot()
        x = np.cos(np.radians(beta)) * np.cos(np.radians(alpha))
        y = np.cos(np.radians(beta)) * np.sin(np.radians(alpha))
        z = np.sin(np.radians(beta))
        self.ax.quiver(0, 0, 0, x, y, z, length=0.8, color="red")
        self.draw()


# --- Ana Arayüz Sınıfı Başlangıcı ---
class UGVInterface(QWidget):
    gyro_data_received = pyqtSignal(str, float, float, float)

    def __init__(self):
        super().__init__()
        self.init_ui()

        self.cap1 = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.timer1 = QTimer()
        self.timer1.timeout.connect(self.update_camera1)
        self.timer1.start(30)

        self.start_camera_stream(2, "192.168.148.186", 9999)
        self.start_camera_stream(3, "192.168.148.12", 9999)

        self.timer2 = QTimer()
        self.timer2.timeout.connect(self.update_camera2)
        self.timer2.start(30)

        self.timer3 = QTimer()
        self.timer3.timeout.connect(self.update_camera3)
        self.timer3.start(30)

        self.gyro_data_received.connect(self.update_gyro_display)
        self.start_websocket_server()

    def paintEvent(self, event):
        painter = QPainter(self)
        if os.path.exists("background.png"):
            pixmap = QPixmap("background.png")
            if not pixmap.isNull():
                painter.drawPixmap(self.rect(), pixmap)
    def init_ui(self):
        self.setWindowTitle("UGV Interface")
        self.setGeometry(0, 0, 1920, 1080)
        self.setStyleSheet("background-color: transparent;")

        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(20, 20, 20, 20)

        left_layout = QVBoxLayout()
        left_layout.setAlignment(Qt.AlignBottom)
        left_layout.setSpacing(5)

        right_layout = QVBoxLayout()
        right_layout.setAlignment(Qt.AlignTop)
        right_layout.setSpacing(10)
        right_layout.addStretch()

        # Radar paneli
        self.radar_panel = RadarWidget()
        right_layout.addWidget(self.radar_panel)

        # Kamera kutuları
        self.camera1 = QLabel()
        self.camera2 = QLabel()
        self.camera3 = QLabel()
        for cam in [self.camera1, self.camera2, self.camera3]:
            cam.setFixedSize(400, 250)
            cam.setStyleSheet("background-color: black; border: none;")
            cam.setAlignment(Qt.AlignCenter)
            left_layout.addWidget(cam)

        # Sensör kutusu
        sensor_frame = QFrame()
        sensor_frame.setStyleSheet("border: 2px solid black; background-color: rgba(255,255,255,150);")
        sensor_frame.setFixedSize(310, 340)
        sensor_layout = QVBoxLayout()

        self.gyro_label = QLabel("Jiroskop: Veriler bekleniyor...")
        self.speed_label = QLabel("Hız: -")
        self.distance_label = QLabel("Mesafe: -")
        for lbl in [self.gyro_label, self.speed_label, self.distance_label]:
            lbl.setStyleSheet("font-size: 13px; padding: 2px;")
            sensor_layout.addWidget(lbl)

        self.gyro_canvas = GyroCanvas()
        self.gyro_canvas.setFixedHeight(200)
        sensor_layout.addWidget(self.gyro_canvas)
        sensor_frame.setLayout(sensor_layout)
        right_layout.addWidget(sensor_frame)
        # D-pad kontrolleri
        control_row = QHBoxLayout()
        control_row.setSpacing(5)

        for title, prefix in [("Araç Kontrol", ""), ("Nişan Kontrol", "aim_")]:
            control_frame = QFrame()
            control_frame.setStyleSheet("border: 2px solid black; background-color: rgba(255,255,255,150);")
            control_frame.setFixedSize(280, 280)
            layout = QVBoxLayout()
            label = QLabel(title)
            label.setAlignment(Qt.AlignCenter)
            layout.addWidget(label)

            grid = QGridLayout()
            grid.setHorizontalSpacing(20)
            grid.setVerticalSpacing(40)
            setattr(self, prefix + "forward_btn", self.create_button("↑"))
            setattr(self, prefix + "left_btn", self.create_button("←"))
            setattr(self, prefix + "right_btn", self.create_button("→"))
            setattr(self, prefix + "backward_btn", self.create_button("↓"))
            setattr(self, prefix + "stop_btn", self.create_button("■"))
            grid.addWidget(getattr(self, prefix + "forward_btn"), 0, 1)
            grid.addWidget(getattr(self, prefix + "left_btn"), 1, 0)
            grid.addWidget(getattr(self, prefix + "stop_btn"), 1, 1)
            grid.addWidget(getattr(self, prefix + "right_btn"), 1, 2)
            grid.addWidget(getattr(self, prefix + "backward_btn"), 2, 1)
            layout.addLayout(grid)
            control_frame.setLayout(layout)
            control_row.addWidget(control_frame)

        right_layout.addLayout(control_row)

        main_layout.addLayout(left_layout)
        main_layout.addStretch()
        main_layout.addLayout(right_layout)
        self.setLayout(main_layout)

    def create_button(self, text, width=60):
        btn = QPushButton(text)
        btn.setFixedSize(width, 60)
        btn.setStyleSheet("background-color: black; color: white; border-radius: 30px;")
        btn.pressed.connect(lambda: btn.setStyleSheet("background-color: red; color: white; border-radius: 30px;"))
        btn.released.connect(lambda: btn.setStyleSheet("background-color: black; color: white; border-radius: 30px;"))
        return btn
    def update_camera1(self):
        ret, frame = self.cap1.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            q_img = QImage(frame.data, w, h, ch * w, QImage.Format_RGB888)
            self.camera1.setPixmap(QPixmap.fromImage(q_img).scaled(400, 250, Qt.KeepAspectRatio))

    def start_camera_stream(self, cam_id, ip, port):
        try:
            cam_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            cam_socket.connect((ip, port))
            setattr(self, f"cam{cam_id}_socket", cam_socket)
            setattr(self, f"cam{cam_id}_data", b'')
            setattr(self, f"cam{cam_id}_payload_size", struct.calcsize(">L"))
        except Exception:
            setattr(self, f"cam{cam_id}_socket", None)
            setattr(self, f"cam{cam_id}_data", b'')

    def update_camera_stream(self, cam_id, target_label):
        try:
            cam_data = getattr(self, f"cam{cam_id}_data", b'')
            cam_socket = getattr(self, f"cam{cam_id}_socket", None)
            payload_size = getattr(self, f"cam{cam_id}_payload_size", 0)
            if cam_socket is None:
                return
            while len(cam_data) < payload_size:
                packet = cam_socket.recv(4096)
                if not packet:
                    return
                cam_data += packet
            packed_msg_size = cam_data[:payload_size]
            cam_data = cam_data[payload_size:]
            msg_size = struct.unpack(">L", packed_msg_size)[0]
            while len(cam_data) < msg_size:
                packet = cam_socket.recv(4096)
                if not packet:
                    return
                cam_data += packet
            frame_data = cam_data[:msg_size]
            cam_data = cam_data[msg_size:]
            setattr(self, f"cam{cam_id}_data", cam_data)
            frame = pickle.loads(frame_data)
            if frame is None or frame.size == 0:
                return
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            q_img = QImage(frame.data, w, h, ch * w, QImage.Format_RGB888)
            target_label.setPixmap(QPixmap.fromImage(q_img).scaled(400, 250, Qt.KeepAspectRatio))
        except:
            pass

    def update_camera2(self):
        self.update_camera_stream(2, self.camera2)

    def update_camera3(self):
        self.update_camera_stream(3, self.camera3)

    def update_gyro_display(self, msg, alpha, beta, gamma):
        self.gyro_label.setText("Jiroskop: " + msg)
        self.gyro_canvas.draw_arrow(alpha, beta, gamma)

    def start_websocket_server(self):
        def run_server():
            asyncio.set_event_loop(asyncio.new_event_loop())
            loop = asyncio.get_event_loop()

            async def handle_connection(websocket, path=None):
                try:
                    async for message in websocket:
                        data = json.loads(message)
                        alpha = float(data.get("alpha", 0))
                        beta = float(data.get("beta", 0))
                        gamma = float(data.get("gamma", 0))
                        msg = f"α: {alpha:.2f}, β: {beta:.2f}, γ: {gamma:.2f}"
                        self.gyro_data_received.emit(msg, alpha, beta, gamma)
                except:
                    pass

            async def server_main():
                async with websockets.serve(handle_connection, "0.0.0.0", 9001):
                    await asyncio.Future()

            loop.run_until_complete(server_main())

        threading.Thread(target=run_server, daemon=True).start()

    def closeEvent(self, event):
        if self.cap1.isOpened():
            self.cap1.release()
        for cam_id in [2, 3]:
            try:
                sock = getattr(self, f"cam{cam_id}_socket", None)
                if sock:
                    sock.close()
            except:
                pass
        event.accept()


# --- Main ---
if __name__ == "__main__":
    try:
        app = QApplication(sys.argv)
        window = UGVInterface()
        window.show()
        sys.exit(app.exec_())
    except Exception as e:
        print("Hata:", e)
        input("Kapatmak için Enter...")


