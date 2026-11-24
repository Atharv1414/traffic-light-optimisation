"""Detection engine: frames in, vehicles per lane out."""

from collections import deque
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from .messages import Box, Frame, LaneCounts
from .scene import CAR_L, CAR_W, lane_polygons

COCO_VEHICLES = (2, 3, 5, 7)  # car, motorcycle, bus, truck


class Detector(Protocol):
    def detect(self, image: np.ndarray) -> list[Box]: ...


class YoloDetector:
    """YOLOv8 through Ultralytics. Works with COCO weights or the locally trained model."""

    def __init__(
        self,
        weights: Path,
        confidence: float = 0.35,
        image_size: int = 640,
        device: str = "cpu",
    ) -> None:
        from ultralytics import YOLO

        if not Path(weights).exists():
            raise FileNotFoundError(f"{weights} not found. Run `traffic-lights train` first.")
        self._model = YOLO(str(weights))
        self._options = {
            "conf": confidence,
            "imgsz": image_size,
            "device": device,
            "verbose": False,
            "max_det": 300,
        }
        names = self._model.names
        self._classes = list(COCO_VEHICLES) if len(names) > 1 else [0]

    def detect(self, image: np.ndarray) -> list[Box]:
        result = self._model.predict(image, classes=self._classes, **self._options)[0]
        return [
            Box(*map(float, xyxy), float(conf))
            for xyxy, conf in zip(
                result.boxes.xyxy.tolist(), result.boxes.conf.tolist(), strict=True
            )
        ]


class ContourDetector:
    """Classical OpenCV baseline for the synthetic camera. Needs no model or PyTorch."""

    def detect(self, image: np.ndarray) -> list[Box]:
        blurred = cv2.GaussianBlur(image, (3, 3), 0)
        road = np.zeros(image.shape[:2], np.uint8)
        for polygon in lane_polygons().values():
            cv2.fillPoly(road, [polygon.astype(np.int32)], 255)
        level = np.median(blurred[road > 0], axis=0)
        difference = np.abs(blurred.astype(np.float32) - level).max(axis=2)
        mask = np.where((difference > 20) & (road > 0), 255, 0).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w * h >= 0.4 * CAR_L * CAR_W:
                boxes.append(Box(x, y, x + w, y + h))
        return boxes


class LaneCounter:
    """Assign boxes to lane polygons, then take a short rolling median to remove flicker."""

    def __init__(self, polygons: dict[str, np.ndarray] | None = None, window: int = 3) -> None:
        self._polygons = polygons or lane_polygons()
        self._history = {lane: deque(maxlen=window) for lane in self._polygons}

    def raw(self, boxes: list[Box]) -> dict[str, int]:
        counts = dict.fromkeys(self._polygons, 0)
        for box in boxes:
            for lane, polygon in self._polygons.items():
                if cv2.pointPolygonTest(polygon, box.centre, False) >= 0:
                    counts[lane] += 1
                    break  # one vehicle, one lane
        return counts

    def update(self, boxes: list[Box]) -> dict[str, int]:
        for lane, n in self.raw(boxes).items():
            self._history[lane].append(n)
        return {lane: int(np.median(h)) for lane, h in self._history.items()}


class DetectionEngine:
    def __init__(self, detector: Detector, counter: LaneCounter | None = None) -> None:
        self._detector = detector
        self._counter = counter or LaneCounter()

    def process(self, frame: Frame) -> LaneCounts:
        counts = self._counter.update(self._detector.detect(frame.image))
        return LaneCounts(frame.sequence, frame.captured_at, counts)
