import datetime
import re
import sys
from pathlib import Path

import cv2
from PyQt5.QtCore import QRect, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from calibration_utils import (
    camera_point_from_pixel,
    camera_to_base,
    load_transformation,
    pose_to_matrix,
)
from camera import SortingCamera
from feature_dialog import FeatureTeachDialog
from recognition import ModelLoadWorker, RecognitionEngine
from robot_online import RobotOnlineClient


class RoiImageLabel(QLabel):
    """Live RGB preview with draggable rectangular ROI selection."""

    selection_changed = pyqtSignal(int, int, int, int)
    selection_cleared = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(420, 320)
        self.setMouseTracking(True)
        self.setStyleSheet(
            "background:#111827; color:#D1D5DB; border-radius:6px;"
        )
        self._source_image = None
        self._selection = None
        self._drag_start = None

    def set_source_image(self, image):
        self._source_image = image.copy()
        self.update()

    def selection(self):
        return self._selection

    def clear_selection(self):
        self._selection = None
        self._drag_start = None
        self.update()
        self.selection_cleared.emit()

    def _image_layout(self):
        if self._source_image is None or self._source_image.isNull():
            return None, 0.0
        image_width = self._source_image.width()
        image_height = self._source_image.height()
        label_width = self.width()
        label_height = self.height()
        if image_width <= 0 or image_height <= 0 or label_width <= 0 or label_height <= 0:
            return None, 0.0
        scale = min(
            label_width / float(image_width),
            label_height / float(image_height),
        )
        draw_width = max(1, int(round(image_width * scale)))
        draw_height = max(1, int(round(image_height * scale)))
        draw_x = int((label_width - draw_width) / 2)
        draw_y = int((label_height - draw_height) / 2)
        return QRect(draw_x, draw_y, draw_width, draw_height), scale

    def _to_image_coord(self, pos):
        rect, scale = self._image_layout()
        if rect is None or scale <= 0:
            return None
        image_x = (pos.x() - rect.x()) / scale
        image_y = (pos.y() - rect.y()) / scale
        image_x = max(0, min(int(round(image_x)), self._source_image.width() - 1))
        image_y = max(0, min(int(round(image_y)), self._source_image.height() - 1))
        return image_x, image_y

    @staticmethod
    def _normalize_selection(point_a, point_b):
        return (
            min(point_a[0], point_b[0]),
            min(point_a[1], point_b[1]),
            max(point_a[0], point_b[0]),
            max(point_a[1], point_b[1]),
        )

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#111827"))

        if self._source_image is None or self._source_image.isNull():
            painter.setPen(QColor("#D1D5DB"))
            painter.drawText(self.rect(), Qt.AlignCenter, "等待相机...")
            painter.end()
            return

        rect, scale = self._image_layout()
        if rect is None:
            painter.end()
            return

        painter.drawImage(rect, self._source_image)
        if self._selection is not None:
            x1, y1, x2, y2 = self._selection
            draw_x1 = int(rect.x() + x1 * scale)
            draw_y1 = int(rect.y() + y1 * scale)
            draw_x2 = int(rect.x() + x2 * scale)
            draw_y2 = int(rect.y() + y2 * scale)
            painter.setPen(QPen(QColor("#22C55E"), 2))
            painter.drawRect(
                QRect(draw_x1, draw_y1, draw_x2 - draw_x1, draw_y2 - draw_y1)
            )
            center_x = int(round((draw_x1 + draw_x2) / 2.0))
            center_y = int(round((draw_y1 + draw_y2) / 2.0))
            painter.setPen(QPen(QColor("#22C55E"), 1))
            painter.drawLine(center_x - 6, center_y, center_x + 6, center_y)
            painter.drawLine(center_x, center_y - 6, center_x, center_y + 6)
        painter.end()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            point = self._to_image_coord(event.pos())
            if point is not None:
                self._drag_start = point
                self._selection = (point[0], point[1], point[0], point[1])
                self.update()
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._drag_start is not None and event.buttons() & Qt.LeftButton:
            point = self._to_image_coord(event.pos())
            if point is not None:
                self._selection = self._normalize_selection(self._drag_start, point)
                self.update()
                event.accept()
                return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self._drag_start is not None:
            point = self._to_image_coord(event.pos())
            if point is not None:
                self._selection = self._normalize_selection(self._drag_start, point)
                self.update()
                self.selection_changed.emit(*self._selection)
            self._drag_start = None
            event.accept()
            return
        super().mouseReleaseEvent(event)


def parse_numbers(text):
    return [float(value) for value in re.findall(r"[-+]?\d+(?:\.\d+)?", text)]


class SortingMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("视觉分拣监控矩阵")
        self.resize(1500, 860)

        self.camera = None
        self.robot = None
        self.engine = RecognitionEngine()
        self.calibration_matrix = None
        self.calibration_path = None
        self.current_target = None
        self.current_depth_m = None
        self.current_roi = None
        self.feature_dialog = None
        self.model_worker = None
        self.current_overlay = None
        self._topmost = False

        self.project_root = Path(__file__).resolve().parent.parent
        self.feature_data_dir = (
            self.project_root / "vision_sorting" / "feature_data"
        )
        self.feature_config_path = self.feature_data_dir / "config.json"
        self.default_model_path = (
            self.project_root
            / "scripts"
            / "runs"
            / "detect"
            / "runs"
            / "train"
            / "fruit_yolov8s_advanced-3"
            / "weights"
            / "best.pt"
        )
        self.default_calib_path = (
            self.project_root
            / "dobot_handeye"
            / "calib_data"
            / "hand_eye_result.yaml"
        )

        self._build_ui()
        self._update_button_states()
        if self.feature_config_path.exists():
            try:
                self.engine.load_feature_config(self.feature_config_path)
                self._log("[系统] 已加载特征示教配置")
            except Exception as exc:
                self._log("[提示] 特征配置未加载: %s" % exc)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._build_control_panel())
        splitter.addWidget(self._build_center_panel())
        splitter.addWidget(self._build_camera_panel())
        splitter.setSizes([300, 850, 250])
        root.addWidget(splitter, 1)

    def _build_control_panel(self):
        panel = QGroupBox("控制指令")
        layout = QVBoxLayout(panel)

        self.ip_input = QLineEdit("192.168.159.1")
        self.port_input = QLineEdit("2001")
        layout.addWidget(QLabel("机器人 IP:"))
        layout.addWidget(self.ip_input)
        layout.addWidget(QLabel("机器人端口:"))
        layout.addWidget(self.port_input)

        self.robot_status = QLabel("● 未连接")
        self.robot_status.setStyleSheet("color:#DC2626; font-weight:bold;")
        layout.addWidget(self.robot_status)

        self.connect_btn = QPushButton("机器人连接")
        self.connect_btn.clicked.connect(self._toggle_robot)
        layout.addWidget(self.connect_btn)

        self.model_btn = QPushButton("加载模型")
        self.model_btn.clicked.connect(self._load_model)
        layout.addWidget(self.model_btn)

        self.feature_btn = QPushButton("特征示教配置")
        self.feature_btn.clicked.connect(self._open_feature_teach)
        layout.addWidget(self.feature_btn)

        self.recognize_btn = QPushButton("单次识别")
        self.recognize_btn.clicked.connect(self._recognize_once)
        layout.addWidget(self.recognize_btn)

        self.clear_roi_btn = QPushButton("清空框选")
        self.clear_roi_btn.clicked.connect(self._clear_roi)
        layout.addWidget(self.clear_roi_btn)

        self.feature_info_label = QLabel("中心 / 深度：未识别")
        self.feature_info_label.setWordWrap(True)
        layout.addWidget(self.feature_info_label)

        self.calib_btn = QPushButton("加载标定")
        self.calib_btn.clicked.connect(self._load_calibration)
        layout.addWidget(self.calib_btn)

        self.transform_btn = QPushButton("执行坐标转换")
        self.transform_btn.clicked.connect(self._coordinate_transform)
        layout.addWidget(self.transform_btn)

        self.grab_btn = QPushButton("执行抓取")
        self.grab_btn.clicked.connect(self._execute_grab)
        layout.addWidget(self.grab_btn)

        layout.addStretch(1)
        return panel

    def _build_center_panel(self):
        center = QWidget()
        layout = QVBoxLayout(center)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        view_group = QGroupBox("视觉矩阵总线")
        view_layout = QHBoxLayout(view_group)

        color_box = QVBoxLayout()
        color_box.addWidget(QLabel("RGB 现场画面"))
        self.color_view = RoiImageLabel()
        self.color_view.selection_changed.connect(self._on_roi_selection)
        self.color_view.selection_cleared.connect(self._on_roi_cleared)
        color_box.addWidget(self.color_view, 1)

        depth_box = QVBoxLayout()
        depth_box.addWidget(QLabel("3D 深度视角"))
        self.depth_view = QLabel("等待相机...")
        self.depth_view.setAlignment(Qt.AlignCenter)
        self.depth_view.setMinimumSize(420, 320)
        self.depth_view.setStyleSheet(
            "background:#111827; color:#D1D5DB; border-radius:6px;"
        )
        depth_box.addWidget(self.depth_view, 1)

        view_layout.addLayout(color_box, 1)
        view_layout.addLayout(depth_box, 1)
        layout.addWidget(view_group, 2)

        log_group = QGroupBox("系统日志")
        log_layout = QVBoxLayout(log_group)
        self.log_text = QPlainTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet(
            "QPlainTextEdit { background:#1E1E1E; color:#D1D5DB;"
            " font-family:Consolas; }"
        )
        log_layout.addWidget(self.log_text, 1)
        self.clear_log_btn = QPushButton("清空日志")
        self.clear_log_btn.clicked.connect(self.log_text.clear)
        log_layout.addWidget(self.clear_log_btn)
        layout.addWidget(log_group, 1)
        return center

    def _build_camera_panel(self):
        panel = QGroupBox("状态与相机控制")
        layout = QVBoxLayout(panel)

        self.fps_label = QLabel("状态：未启动")
        layout.addWidget(self.fps_label)

        self.close_camera_btn = QPushButton("启动相机")
        self.close_camera_btn.clicked.connect(self._toggle_camera)
        layout.addWidget(self.close_camera_btn)

        self.topmost_btn = QPushButton("界面置顶")
        self.topmost_btn.clicked.connect(self._toggle_topmost)
        layout.addWidget(self.topmost_btn)
        layout.addStretch(1)
        return panel

    def _toggle_robot(self):
        if self.robot is not None and self.robot.connected:
            self.robot.disconnect()
            self.robot = None
            self.connect_btn.setText("机器人连接")
            self.robot_status.setText("● 未连接")
            self.robot_status.setStyleSheet(
                "color:#DC2626; font-weight:bold;"
            )
            self._log("[网络] 已断开机器人在线服务端")
            return

        try:
            port = int(self.port_input.text())
        except ValueError:
            self._log("[错误] 端口号无效")
            return

        client = RobotOnlineClient(self.ip_input.text(), port)
        try:
            client.connect()
        except OSError as exc:
            self._log("[网络] 连接失败: %s" % exc)
            return

        self.robot = client
        self.connect_btn.setText("断开机器人")
        self.robot_status.setText("● 已连接")
        self.robot_status.setStyleSheet(
            "color:#16A34A; font-weight:bold;"
        )
        self._log("[网络] 机械臂在线服务端连接成功")

    def _load_model(self):
        if self.model_worker is not None and self.model_worker.isRunning():
            return
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择 YOLO 模型",
            str(self.default_model_path.parent)
            if self.default_model_path.parent.exists()
            else str(self.project_root),
            "PyTorch 模型 (*.pt)",
        )
        if not path:
            return
        self.model_btn.setText("模型加载中...")
        self.model_btn.setEnabled(False)
        self._log("[系统] 正在后台加载模型: %s" % path)
        self.model_worker = ModelLoadWorker(self.engine, path)
        self.model_worker.finished.connect(self._on_model_loaded)
        self.model_worker.start()

    def _on_model_loaded(self, success, message):
        self.model_btn.setEnabled(True)
        if success:
            self.model_btn.setText("已加载模型")
            self._log("[系统] 模型加载完成: %s" % message)
        else:
            self.model_btn.setText("加载模型")
            self._log("[错误] 模型加载失败: %s" % message)
        self._update_button_states()

    def _load_calibration(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择标定文件",
            str(self.default_calib_path.parent)
            if self.default_calib_path.parent.exists()
            else str(self.project_root),
            "YAML 文件 (*.yaml *.yml)",
        )
        if not path:
            return
        try:
            self.calibration_matrix = load_transformation(path)
        except Exception as exc:
            self._log("[错误] 标定文件加载失败: %s" % exc)
            return
        self.calibration_path = path
        self._log("[系统] 标定矩阵已加载: %s" % path)
        self._update_button_states()

    def _toggle_camera(self):
        if self.camera is not None and self.camera.isRunning():
            self.camera.stop()
            self.camera = None
            self.close_camera_btn.setText("启动相机")
            self.fps_label.setText("状态：已停止")
            return

        self.camera = SortingCamera()
        self.camera.color_ready.connect(self._update_color)
        self.camera.depth_ready.connect(self._update_depth)
        self.camera.status_changed.connect(self._on_camera_status)
        self.camera.error_signal.connect(self._log)
        self.camera.fps_signal.connect(self._on_fps)
        self.camera.start()
        self.close_camera_btn.setText("关闭相机")

    def _open_feature_teach(self):
        color = self.camera.latest_color() if self.camera is not None else None
        if color is None:
            self._log("[提示] 相机尚未提供实时画面")
            return
        self.feature_dialog = FeatureTeachDialog(self)
        self.feature_dialog.set_image(color)
        self.feature_dialog.sync_callback = self.camera.latest_color
        if self.feature_dialog.exec_():
            try:
                self.feature_dialog.save_config(self.feature_config_path)
                self.engine.load_feature_config(self.feature_config_path)
                self._log(
                    "[系统] 特征示教配置已保存: %s" % self.feature_config_path
                )
            except Exception as exc:
                self._log("[错误] 特征配置保存失败: %s" % exc)

    def _recognize_once(self):
        if self.camera is None:
            self._log("[错误] 相机未启动")
            return
        color = self.camera.latest_color()
        if color is None:
            self._log("[错误] 未获取到画面")
            return

        label, center, score, best = self.engine.recognize_with_feature(color)
        if center[0] is not None and label != "!!!":
            box = best.get("box") if best else None
            self.current_overlay = {"label": label, "box": box}
            depth_m = self.camera.get_depth_at(center[0], center[1])
            self.current_depth_m = depth_m
            depth_mm = depth_m * 1000.0 if depth_m > 0 else 0.0
            self.feature_info_label.setText(
                "中心 (%.1f, %.1f) | 深度 %.1f mm"
                % (center[0], center[1], depth_mm)
            )
            self._log(
                "[识别] %s | 匹配得分 %.1f | 中心 (%.1f, %.1f) | 深度 %.1f mm"
                % (label, score, center[0], center[1], depth_mm)
            )
            print(label, flush=True)
            return label, center

        if center[0] is None:
            self.current_overlay = None
            self.current_depth_m = None
            self.feature_info_label.setText("中心 / 深度：无有效目标")
            self._log("[识别] !!! 未识别到目标，且未找到有效轮廓")
            print("!!!", flush=True)
            return "!!!", None
        self.current_overlay = None
        depth_m = self.camera.get_depth_at(center[0], center[1])
        self.current_depth_m = depth_m
        depth_mm = depth_m * 1000.0 if depth_m > 0 else 0.0
        self.feature_info_label.setText(
            "中心 (%.1f, %.1f) | 深度 %.1f mm"
            % (center[0], center[1], depth_mm)
        )
        self._log(
            "[识别] !!! 未匹配到特征，使用轮廓中心 (%.1f, %.1f) | 深度 %.1f mm"
            % (center[0], center[1], depth_mm)
        )
        print("!!!", flush=True)
        return "!!!", center

    def _clear_roi(self):
        self.color_view.clear_selection()

    def _on_roi_cleared(self):
        self.current_roi = None
        self.feature_info_label.setText("中心 / 深度：未识别")

    def _on_roi_selection(self, x1, y1, x2, y2):
        self.current_roi = (x1, y1, x2, y2)
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        if self.camera is None:
            self.feature_info_label.setText(
                "中心 (%.1f, %.1f) | 深度：相机未启动"
                % (center_x, center_y)
            )
            return

        depth_m = self.camera.get_depth_at(center_x, center_y)
        self.current_depth_m = depth_m
        depth_mm = depth_m * 1000.0 if depth_m > 0 else 0.0
        self.feature_info_label.setText(
            "框选中心 (%.1f, %.1f) | 深度 %.1f mm"
            % (center_x, center_y, depth_mm)
        )
        self._log(
            "[框选] 区域中心 (%.1f, %.1f) | 深度 %.1f mm"
            % (center_x, center_y, depth_mm)
        )

    def _coordinate_transform(self):
        if self.robot is None or not self.robot.connected:
            self._log("[错误] 机器人未连接")
            return
        if self.calibration_matrix is None:
            self._log("[错误] 请先加载标定")
            return
        if self.camera is None:
            self._log("[错误] 相机未启动")
            return

        if self.current_roi is not None:
            x1, y1, x2, y2 = self.current_roi
            label = "!!!"
            center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            self.current_overlay = None
            self._log("[框选] 使用手动框选中心执行坐标转换")
        else:
            result = self._recognize_once()
            if result is None:
                return
            label, center = result
        if center is None:
            return

        depth = self.camera.get_depth_at(center[0], center[1])
        if depth <= 0.0:
            self.current_depth_m = depth
            self.feature_info_label.setText(
                "中心 (%.1f, %.1f) | 深度无效" % (center[0], center[1])
            )
            self._log("[错误] 目标点深度无效")
            return
        self.current_depth_m = depth
        self.feature_info_label.setText(
            "中心 (%.1f, %.1f) | 深度 %.1f mm"
            % (center[0], center[1], depth * 1000.0)
        )

        intrinsics = self.camera.get_color_intrinsics()
        if intrinsics is None:
            self._log("[错误] 相机内参不可用")
            return

        try:
            response = self.robot.send_and_receive("getpose")
        except OSError as exc:
            self._log("[网络] getpose 通信失败: %s" % exc)
            return
        numbers = parse_numbers(response)
        if len(numbers) < 6:
            self._log("[网络] getpose 返回数据不完整: %s" % response)
            return

        robot_pose = numbers[:6]
        point_camera = camera_point_from_pixel(
            center[0], center[1], depth, intrinsics
        )
        gripper_to_base = pose_to_matrix(robot_pose)
        base_xyz = camera_to_base(
            point_camera, self.calibration_matrix, gripper_to_base
        )

        self.current_target = {
            "label": label,
            "base": [float(value) for value in base_xyz],
        }
        self._log(
            "[转换成功] 绝对抓取点 -> X:%.1f Y:%.1f Z:%.1f"
            % tuple(self.current_target["base"])
        )
        self._update_button_states()

    def _execute_grab(self):
        if self.robot is None or not self.robot.connected:
            self._log("[错误] 机器人未连接")
            return
        if self.current_target is None:
            self._log("[错误] 请先执行坐标转换")
            return
        label = (self.current_target.get("label") or "").strip().lower()
        if label in ("", "!!!"):
            label, ok = QInputDialog.getText(
                self,
                "手动输入类别",
                "未识别到具体类别，请输入物料类别：",
                QLineEdit.Normal,
                "",
            )
            label = label.strip().lower()
            if not ok or not label:
                self._log("[提示] 未输入类别，已取消抓取发送")
                return
            self.current_target["label"] = label
        x, y, z = self.current_target["base"]
        command = "%s,%.3f,%.3f,%.3f" % (label, x, y, z)
        try:
            self.robot.send_and_receive(command)
        except OSError as exc:
            self._log("[网络] 抓取指令发送失败: %s" % exc)
            return
        self._log("[抓取] 已发送: %s" % command)

    def _update_color(self, image):
        color_bgr = self.camera.latest_color() if self.camera is not None else None
        if color_bgr is None:
            return
        display = self._draw_overlay(color_bgr)
        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb.shape
        qimage = QImage(
            rgb.data,
            width,
            height,
            channels * width,
            QImage.Format_RGB888,
        ).copy()
        self.color_view.set_source_image(qimage)

    def _draw_overlay(self, color_bgr):
        display = color_bgr.copy()
        if self.current_overlay is None:
            return display
        box = self.current_overlay.get("box")
        label = self.current_overlay.get("label", "")
        if box is not None:
            x1, y1, x2, y2 = [int(value) for value in box]
            cv2.rectangle(display, (x1, y1), (x2, y2), (255, 200, 0), 2)
            if label:
                cv2.putText(
                    display,
                    str(label),
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 200, 0),
                    2,
                )
        return display

    def _update_depth(self, image):
        self.depth_view.setPixmap(
            QPixmap.fromImage(image).scaled(
                self.depth_view.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )

    def _on_fps(self, fps):
        self.fps_label.setText("状态：传输中 (%.0f FPS)" % fps)

    def _on_camera_status(self, status):
        self.fps_label.setText("状态：" + status)

    def _toggle_topmost(self):
        self._topmost = not self._topmost
        self.setWindowFlag(Qt.WindowStaysOnTopHint, self._topmost)
        self.show()
        self.topmost_btn.setText("取消置顶" if self._topmost else "界面置顶")

    def _update_button_states(self):
        model_ready = self.engine.loaded
        calib_ready = self.calibration_matrix is not None
        self.transform_btn.setEnabled(model_ready and calib_ready)
        self.grab_btn.setEnabled(self.current_target is not None)

    def _log(self, message):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.log_text.appendPlainText("[%s] %s" % (timestamp, message))

    def closeEvent(self, event):
        if self.model_worker is not None and self.model_worker.isRunning():
            self.model_worker.wait(2000)
        if self.camera is not None:
            self.camera.stop()
        if self.robot is not None:
            self.robot.disconnect()
        super().closeEvent(event)
