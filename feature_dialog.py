import json
from pathlib import Path

import cv2
import numpy as np
from PyQt5.QtCore import QPoint, Qt, pyqtSignal
from PyQt5.QtGui import QImage, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)


class TeachImageLabel(QLabel):
    roi_finished = pyqtSignal(tuple)

    def __init__(self):
        super().__init__()
        self.setMinimumSize(720, 420)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background:#111827; color:#D1D5DB;")
        self.setText("等待框选...")
        self._image_bgr = None
        self._display_pixmap = None
        self._scale = 1.0
        self._offset_x = 0
        self._offset_y = 0
        self._start = None
        self._current = None
        self._finalized = False
        self._center_display = None
        self._size_display = None
        self._center_original = None
        self._size_original = None
        self._angle = 0.0
        self._dragging_handle = False
        self._contours = []

    def set_cv_image(self, color_bgr):
        self._image_bgr = color_bgr
        rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb.shape
        qimage = QImage(
            rgb.data,
            width,
            height,
            channels * width,
            QImage.Format_RGB888,
        ).copy()
        self._display_pixmap = QPixmap.fromImage(qimage).scaled(
            self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.setPixmap(self._display_pixmap)
        self._scale = min(
            self.width() / width,
            self.height() / height,
        )
        self._offset_x = (self.width() - width * self._scale) / 2.0
        self._offset_y = (self.height() - height * self._scale) / 2.0
        self._start = None
        self._current = None
        self._finalized = False
        self._center_display = None
        self._size_display = None
        self._center_original = None
        self._size_original = None
        self._angle = 0.0
        self._dragging_handle = False
        self._contours = []

    @property
    def roi(self):
        if self._center_original is None or self._size_original is None:
            return None
        return (self._center_original, self._size_original, self._angle)

    def _to_original(self, pos):
        x = (pos.x() - self._offset_x) / self._scale
        y = (pos.y() - self._offset_y) / self._scale
        h, w = self._image_bgr.shape[:2]
        return max(0, min(w - 1, int(x))), max(0, min(h - 1, int(y)))

    def original_to_display(self, point):
        x, y = point
        return (
            self._offset_x + float(x) * self._scale,
            self._offset_y + float(y) * self._scale,
        )

    def set_contours(self, display_contours):
        self._contours = display_contours
        self.update()

    def _display_handle(self):
        if self._center_display is None or self._size_display is None:
            return None
        box = cv2.boxPoints(
            (
                (self._center_display[0], self._center_display[1]),
                self._size_display,
                self._angle,
            )
        )
        return box[0]

    def _near_handle(self, pos, threshold=12):
        handle = self._display_handle()
        if handle is None:
            return False
        return abs(pos.x() - handle[0]) <= threshold and abs(
            pos.y() - handle[1]
        ) <= threshold

    def mousePressEvent(self, event):
        if self._display_pixmap is None:
            return
        if not self._finalized:
            if self._start is None:
                self._start = event.pos()
                self._current = event.pos()
            else:
                x1, y1 = self._to_original(self._start)
                x2, y2 = self._to_original(event.pos())
                left = min(x1, x2)
                top = min(y1, y2)
                right = max(x1, x2)
                bottom = max(y1, y2)
                self._center_original = (
                    (left + right) / 2.0,
                    (top + bottom) / 2.0,
                )
                self._size_original = (
                    float(right - left),
                    float(bottom - top),
                )
                self._center_display = (
                    (self._start.x() + event.pos().x()) / 2.0,
                    (self._start.y() + event.pos().y()) / 2.0,
                )
                self._size_display = (
                    float(abs(event.pos().x() - self._start.x())),
                    float(abs(event.pos().y() - self._start.y())),
                )
                self._angle = 0.0
                self._finalized = True
                self._current = event.pos()
                self.roi_finished.emit(
                    (
                        self._center_original,
                        self._size_original,
                        self._angle,
                    )
                )
                self.update()
            return

        if self._near_handle(event.pos()):
            self._dragging_handle = True

    def mouseMoveEvent(self, event):
        if self._start is not None and not self._finalized:
            self._current = event.pos()
            self.update()
            return

        if self._dragging_handle and self._center_display is not None:
            dx = event.pos().x() - self._center_display[0]
            dy = event.pos().y() - self._center_display[1]
            self._angle = float(np.degrees(np.arctan2(dy, dx)))
            self.update()

    def mouseReleaseEvent(self, event):
        self._dragging_handle = False

    def _box_points_display(self):
        if self._center_display is None or self._size_display is None:
            return None
        return cv2.boxPoints(
            (
                (self._center_display[0], self._center_display[1]),
                self._size_display,
                self._angle,
            )
        ).astype(int)

    def paintEvent(self, event):
        super().paintEvent(event)
        if self._start is None or self._current is None:
            return

        painter = QPainter(self)
        for contour in self._contours:
            painter.setPen(QPen(Qt.red, 2))
            points = [QPoint(int(point[0][0]), int(point[0][1])) for point in contour]
            painter.drawPolyline(*points)

        if not self._finalized:
            painter.setPen(QPen(Qt.green, 2))
            x1, y1 = self._to_original(self._start)
            x2, y2 = self._to_original(self._current)
            painter.drawRect(
                self._start.x(),
                self._start.y(),
                self._current.x() - self._start.x(),
                self._current.y() - self._start.y(),
            )
        else:
            points = self._box_points_display()
            if points is not None:
                painter.setPen(QPen(Qt.green, 2))
                painter.drawPolygon(
                    *[QPoint(int(point[0]), int(point[1])) for point in points]
                )
            handle = self._display_handle()
            if handle is not None:
                painter.setPen(QPen(Qt.red, 4))
                painter.drawEllipse(
                    QPoint(int(handle[0]), int(handle[1])),
                    6,
                    6,
                )
        painter.end()

    def clear_roi(self):
        self._start = None
        self._current = None
        self._finalized = False
        self._center_display = None
        self._size_display = None
        self._center_original = None
        self._size_original = None
        self._angle = 0.0
        self._contours = []
        self.update()


class FeatureTeachDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("特征示教与掩膜编辑器")
        self.resize(1100, 620)
        self.template_bgr = None
        self.preview_bgr = None
        self.mask = None
        self.sync_callback = None
        self._build_ui()

    def _build_ui(self):
        root = QHBoxLayout(self)
        self.view = TeachImageLabel()
        root.addWidget(self.view, 3)

        panel = QWidget()
        panel_layout = QVBoxLayout(panel)
        panel_layout.addWidget(QLabel("操作指南："))
        panel_layout.addWidget(QLabel("1. 框选需要检索的物体轮廓"))
        panel_layout.addWidget(QLabel("2. 使用特征橡皮擦剔除背景干扰点"))
        panel_layout.addWidget(QLabel("3. 调节匹配分数，保存配置。"))

        self.roi_btn = QPushButton("开始框选特征 (ROI)")
        self.erase_btn = QPushButton("特征橡皮擦")
        self.extract_btn = QPushButton("提取特征")
        self.sync_btn = QPushButton("同步最新实况图")
        self.clear_btn = QPushButton("清空当前框选")
        panel_layout.addWidget(self.roi_btn)
        panel_layout.addWidget(self.erase_btn)
        panel_layout.addWidget(self.extract_btn)
        panel_layout.addWidget(self.sync_btn)
        panel_layout.addWidget(self.clear_btn)

        panel_layout.addWidget(QLabel("特征匹配及格分数："))
        self.score_label = QLabel("80分")
        self.score_slider = QSlider(Qt.Horizontal)
        self.score_slider.setRange(0, 100)
        self.score_slider.setValue(80)
        self.score_slider.valueChanged.connect(
            lambda value: self.score_label.setText("%d分" % value)
        )
        panel_layout.addWidget(self.score_label)
        panel_layout.addWidget(self.score_slider)

        self.model_label = QLabel("当前分类模型：YOLOv8 模型就绪")
        panel_layout.addWidget(self.model_label)

        self.preview_label = QLabel("等待框选...")
        self.preview_label.setMinimumSize(220, 160)
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setStyleSheet(
            "background:#111827; color:#D1D5DB;"
        )
        panel_layout.addWidget(QLabel("特征切片预览"))
        panel_layout.addWidget(self.preview_label)

        self.cancel_btn = QPushButton("取消")
        self.save_btn = QPushButton("确定保存配置")
        panel_layout.addWidget(self.cancel_btn)
        panel_layout.addWidget(self.save_btn)
        panel_layout.addStretch(1)
        root.addWidget(panel, 1)

        self.cancel_btn.clicked.connect(self.reject)
        self.save_btn.clicked.connect(self.accept)
        self.clear_btn.clicked.connect(self._clear_roi)
        self.extract_btn.clicked.connect(self._extract_feature)
        self.sync_btn.clicked.connect(self._sync_latest)
        self.view.roi_finished.connect(self._on_roi_finished)

    def set_image(self, color_bgr):
        self.view.set_cv_image(color_bgr)

    def _clear_roi(self):
        self.view.clear_roi()
        self.template_bgr = None
        self.preview_bgr = None
        self.mask = None
        self.preview_label.setText("等待框选...")

    def _sync_latest(self):
        if self.sync_callback is None:
            return
        frame = self.sync_callback()
        if frame is not None:
            self.set_image(frame)
            if self.view.roi is not None:
                self._extract_feature()

    def _on_roi_finished(self, roi):
        pass

    def _extract_feature(self):
        if self.view._image_bgr is None or self.view.roi is None:
            return
        (cx, cy), (width, height), angle = self.view.roi
        points = cv2.boxPoints(
            ((float(cx), float(cy)), (float(width), float(height)), float(angle))
        )
        x, y, w, h = cv2.boundingRect(points)
        x = max(0, x)
        y = max(0, y)
        w = min(self.view._image_bgr.shape[1] - x, w)
        h = min(self.view._image_bgr.shape[0] - y, h)
        crop = self.view._image_bgr[y : y + h, x : x + w].copy()

        mask = np.full(crop.shape[:2], cv2.GC_BGD, np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        local_center = (int(cx - x), int(cy - y))
        axes = (max(2, int(width / 2)), max(2, int(height / 2)))
        cv2.ellipse(
            mask,
            local_center,
            axes,
            float(angle),
            0,
            360,
            cv2.GC_PR_FGD,
            -1,
        )
        cv2.grabCut(
            crop,
            mask,
            None,
            bgd_model,
            fgd_model,
            5,
            cv2.GC_INIT_WITH_MASK,
        )
        binary_mask = np.where(
            (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
            255,
            0,
        ).astype("uint8")
        masked = cv2.bitwise_and(crop, crop, mask=binary_mask)
        self.template_bgr = masked.copy()
        self.mask = binary_mask

        contours, _ = cv2.findContours(
            binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        display = crop.copy()
        cv2.drawContours(display, contours, -1, (0, 0, 255), 2)
        self.preview_bgr = display

        display_contours = []
        for contour in contours:
            mapped = []
            for point in contour:
                original_x = x + int(point[0][0])
                original_y = y + int(point[0][1])
                mapped.append(
                    [
                        self.view.original_to_display(
                            (original_x, original_y)
                        )
                    ]
                )
            display_contours.append(np.array(mapped, dtype=np.int32).reshape(-1, 1, 2))
        self.view.set_contours(display_contours)
        self._update_preview()

    def _update_preview(self):
        if self.preview_bgr is None:
            return
        rgb = cv2.cvtColor(self.preview_bgr, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb.shape
        image = QImage(
            rgb.data,
            width,
            height,
            channels * width,
            QImage.Format_RGB888,
        ).copy()
        self.preview_label.setPixmap(
            QPixmap.fromImage(image).scaled(
                self.preview_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )

    def get_config(self):
        return {
            "roi": self.view.roi,
            "score": self.score_slider.value(),
        }

    def save_config(self, path):
        config = self.get_config()
        if self.view.roi is None:
            raise ValueError("尚未完成 ROI 框选")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(config, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if self.template_bgr is not None:
            cv2.imwrite(str(path.parent / "template.png"), self.template_bgr)
        if self.mask is not None:
            cv2.imwrite(str(path.parent / "mask.png"), self.mask)
