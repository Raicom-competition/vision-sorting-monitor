import threading
import time

import cv2
import numpy as np
import pyrealsense2 as rs
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QImage


class SortingCamera(QThread):
    color_ready = pyqtSignal(QImage)
    depth_ready = pyqtSignal(QImage)
    status_changed = pyqtSignal(str)
    error_signal = pyqtSignal(str)
    fps_signal = pyqtSignal(float)

    def __init__(self, width=1280, height=720, fps=30, parent=None):
        super().__init__(parent)
        self.width = width
        self.height = height
        self.fps = fps
        self._running = False
        self._pipeline = None
        self._align = None
        self._color_sensor = None
        self._color_bgr = None
        self._depth_frame = None
        self._color_intrinsics = None
        self._lock = threading.Lock()

    def stop(self):
        self._running = False
        self.wait(3000)

    def latest_color(self):
        with self._lock:
            return None if self._color_bgr is None else self._color_bgr.copy()

    def get_depth_at(self, x, y):
        with self._lock:
            if self._depth_frame is None:
                return 0.0
            return float(self._depth_frame.get_distance(int(x), int(y)))

    def get_color_intrinsics(self):
        with self._lock:
            return self._color_intrinsics

    def _to_qimage(self, bgr):
        rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        height, width, channels = rgb.shape
        return QImage(
            rgb.data,
            width,
            height,
            channels * width,
            QImage.Format_RGB888,
        ).copy()

    def run(self):
        self._running = True
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(
            rs.stream.color,
            self.width,
            self.height,
            rs.format.bgr8,
            self.fps,
        )
        config.enable_stream(
            rs.stream.depth,
            640,
            480,
            rs.format.z16,
            self.fps,
        )

        try:
            profile = pipeline.start(config)
        except Exception as exc:
            self.error_signal.emit("相机启动失败: %s" % exc)
            return

        self._pipeline = pipeline
        self._align = rs.align(rs.stream.color)
        color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        self._color_intrinsics = color_stream.get_intrinsics()

        try:
            device = profile.get_device()
            self._color_sensor = next(
                (
                    sensor
                    for sensor in device.query_sensors()
                    if sensor.get_info(rs.camera_info.name) == "RGB Camera"
                ),
                None,
            )
            if self._color_sensor is not None:
                self._color_sensor.set_option(rs.option.enable_auto_exposure, 1.0)
        except Exception:
            self._color_sensor = None

        self.status_changed.emit("相机已开启")
        frame_count = 0
        last_fps_time = time.time()

        try:
            while self._running:
                try:
                    frames = pipeline.wait_for_frames(timeout_ms=1000)
                except RuntimeError as exc:
                    self.error_signal.emit("取流异常: %s" % exc)
                    break
                if not frames:
                    continue

                aligned = self._align.process(frames)
                color_frame = aligned.get_color_frame()
                depth_frame = aligned.get_depth_frame()
                if not color_frame or not depth_frame:
                    continue

                color_bgr = np.asanyarray(color_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())
                depth_vis = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth_image, alpha=0.03),
                    cv2.COLORMAP_JET,
                )

                with self._lock:
                    self._color_bgr = color_bgr.copy()
                    self._depth_frame = depth_frame

                self.color_ready.emit(self._to_qimage(color_bgr))
                self.depth_ready.emit(self._to_qimage(depth_vis))

                frame_count += 1
                now = time.time()
                if now - last_fps_time >= 1.0:
                    fps_value = frame_count / (now - last_fps_time)
                    frame_count = 0
                    last_fps_time = now
                    self.fps_signal.emit(fps_value)
        finally:
            pipeline.stop()
            self._pipeline = None
            with self._lock:
                self._color_bgr = None
                self._depth_frame = None
            self.status_changed.emit("相机已关闭")
