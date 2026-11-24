"""Train YOLOv8n from scratch on labelled frames from the synthetic camera (no downloads)."""

import os
import shutil
from pathlib import Path

import cv2
import numpy as np

from .messages import LANES
from .scene import SIZE, VISIBLE, render


def write_dataset(root: Path, train: int, val: int, seed: int = 0) -> Path:
    root = root.resolve()
    for split, n, offset in (("train", train, 0), ("val", val, 1)):
        rng = np.random.default_rng([seed, offset])  # separate streams: no leakage
        for kind in ("images", "labels"):
            shutil.rmtree(root / kind / split, ignore_errors=True)
            (root / kind / split).mkdir(parents=True)
        for i in range(n):
            top = int(rng.integers(0, VISIBLE + 1))
            counts = {lane: int(rng.integers(0, top + 1)) for lane in LANES}
            image, boxes = render(counts, rng)
            cv2.imwrite(str(root / "images" / split / f"{i:05}.png"), image)
            rows = [
                f"0 {b.centre[0] / SIZE:.6f} {b.centre[1] / SIZE:.6f} "
                f"{(b.x2 - b.x1) / SIZE:.6f} {(b.y2 - b.y1) / SIZE:.6f}"
                for b in boxes
            ]
            (root / "labels" / split / f"{i:05}.txt").write_text("\n".join(rows))
    data = root / "data.yaml"
    data.write_text(f"path: {root}\ntrain: images/train\nval: images/val\nnames:\n  0: vehicle\n")
    return data


def train(
    dataset: Path, output: Path, images: int, epochs: int, device: str, seed: int = 0
) -> dict[str, float]:
    os.environ.setdefault("YOLO_OFFLINE", "true")
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")
    from ultralytics import YOLO

    data = write_dataset(dataset, images, max(20, images // 5), seed)
    model = YOLO("yolov8n.yaml")
    model.train(
        data=str(data),
        epochs=epochs,
        imgsz=SIZE,
        batch=8,
        device=device,
        seed=seed,
        amp=False,
        plots=False,
        workers=0,
        pretrained=False,
        project=str(dataset.resolve() / "runs"),
        name="train",
        exist_ok=True,
        verbose=False,
    )
    best = Path(model.trainer.best)
    shutil.copy(best, output)
    box = (
        YOLO(str(best))
        .val(data=str(data), imgsz=SIZE, device="cpu", plots=False, verbose=False, workers=0)
        .box
    )
    return {
        "precision": float(box.mp),
        "recall": float(box.mr),
        "mAP50": float(box.map50),
        "mAP50-95": float(box.map),
    }
