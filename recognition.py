import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal


class ModelLoadWorker(QThread):
    finished = pyqtSignal(bool, str)

    def __init__(self, engine, model_path, parent=None):
        super().__init__(parent)
        self.engine = engine
        self.model_path = model_path

    def run(self):
        try:
            self.engine.load_model(self.model_path)
        except Exception as exc:
            self.finished.emit(False, str(exc))
            return
        self.finished.emit(True, str(self.model_path))


class RecognitionEngine:
    def __init__(self):
        self.model = None
        self.model_path = None
        self.class_names = []
        self.feature_config = None
        self.feature_template = None
        self.feature_mask = None
        self.feature_threshold = 80

    def load_model(self, model_path):
        from ultralytics import YOLO

        self.model = YOLO(str(model_path))
        self.model_path = str(model_path)
        self.class_names = list(self.model.names.values())

    @property
    def loaded(self):
        return self.model is not None

    def load_feature_config(self, config_path):
        import json
        from pathlib import Path

        config_path = Path(config_path)
        config = json.loads(config_path.read_text(encoding="utf-8"))
        template_path = config_path.parent / "template.png"
        mask_path = config_path.parent / "mask.png"
        if not template_path.exists():
            raise FileNotFoundError("特征模板图片不存在: %s" % template_path)

        template = cv2.imread(str(template_path), cv2.IMREAD_COLOR)
        mask = None
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        self.feature_config = config
        self.feature_template = template
        self.feature_mask = mask
        self.feature_threshold = int(config.get("score", 80))

    def detect(self, color_bgr, conf=0.35):
        if self.model is None:
            return []
        results = self.model.predict(
            color_bgr,
            conf=conf,
            verbose=False,
            device="0" if _cuda_available() else "cpu",
        )
        detections = []
        if not results:
            return detections
        result = results[0]
        if result.boxes is None:
            return detections
        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        confs = result.boxes.conf.cpu().numpy()
        for box, class_id, confidence in zip(boxes, classes, confs):
            x1, y1, x2, y2 = [float(value) for value in box]
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            name = (
                self.class_names[class_id]
                if class_id < len(self.class_names)
                else str(class_id)
            )
            detections.append(
                {
                    "label": name,
                    "confidence": float(confidence),
                    "box": (x1, y1, x2, y2),
                    "center": (cx, cy),
                }
            )
        return detections

    def contour_center(self, color_bgr):
        gray = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None, None
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < 200:
            return None, None
        moments = cv2.moments(largest)
        if moments["m00"] == 0:
            return None, None
        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]
        return int(cx), int(cy)

    def match_feature(self, color_bgr, box):
        if self.feature_template is None:
            return 0.0
        x1, y1, x2, y2 = [int(value) for value in box]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(color_bgr.shape[1], x2)
        y2 = min(color_bgr.shape[0], y2)
        if x2 - x1 < 8 or y2 - y1 < 8:
            return 0.0

        crop = color_bgr[y1:y2, x1:x2]
        crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        template_gray = cv2.cvtColor(self.feature_template, cv2.COLOR_BGR2GRAY)
        if self.feature_mask is not None:
            mask = cv2.resize(
                self.feature_mask,
                (template_gray.shape[1], template_gray.shape[0]),
            )
            crop_gray = cv2.resize(
                crop_gray,
                (template_gray.shape[1], template_gray.shape[0]),
            )
            crop_gray = cv2.bitwise_and(crop_gray, crop_gray, mask=mask)
            template_gray = cv2.bitwise_and(
                template_gray, template_gray, mask=mask
            )
        else:
            crop_gray = cv2.resize(
                crop_gray,
                (template_gray.shape[1], template_gray.shape[0]),
            )

        result = cv2.matchTemplate(
            template_gray, crop_gray, cv2.TM_CCOEFF_NORMED
        )
        _, max_value, _, _ = cv2.minMaxLoc(result)
        return float(max_value * 100.0)

    def recognize_with_feature(self, color_bgr):
        detections = self.detect(color_bgr)
        if self.feature_template is not None and detections:
            scored = []
            for detection in detections:
                score = self.match_feature(color_bgr, detection["box"])
                detection["feature_score"] = score
                scored.append(detection)
            scored.sort(key=lambda item: item["feature_score"], reverse=True)
            best = scored[0]
            if best["feature_score"] >= self.feature_threshold:
                return (
                    best["label"],
                    best["center"],
                    best["feature_score"],
                    best,
                )

            center = self.contour_center(color_bgr)
            return "!!!", center, best["feature_score"], best

        if detections:
            first = detections[0]
            return (
                first["label"],
                first["center"],
                first["confidence"] * 100.0,
                first,
            )

        center = self.contour_center(color_bgr)
        return "!!!", center, 0.0, None


def _cuda_available():
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False
