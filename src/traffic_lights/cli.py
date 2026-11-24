import argparse
import json
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

from .config import ConfigError, load_config
from .controller import SignalController
from .detection import ContourDetector, DetectionEngine, Detector, LaneCounter, YoloDetector
from .evaluate import camera_limit, compare, markdown, simulate, vision_sensor
from .messages import LANES, Mode, SignalState
from .pipeline import Pipeline
from .scene import VISIBLE, render
from .simulation import SCENARIOS

CONFIG = Path("config/junction.yaml")


def make_detector(args: argparse.Namespace) -> Detector:
    if args.detector == "contour":
        return ContourDetector()
    return YoloDetector(args.weights, device=args.device)


def synthetic_camera(seconds: float, fps: float) -> Iterator[np.ndarray]:
    rng = np.random.default_rng(0)
    queues = dict.fromkeys(LANES, 0)
    for _ in range(int(seconds * fps)):
        lane = LANES[int(rng.integers(len(LANES)))]
        queues[lane] = int(np.clip(queues[lane] + rng.integers(-2, 4), 0, VISIBLE))
        yield render(queues, rng)[0]
        time.sleep(1 / fps)


def video_file(path: str) -> Iterator[np.ndarray]:
    capture = cv2.VideoCapture(int(path) if path.isdigit() else path)
    if not capture.isOpened():
        raise OSError(f"cannot open video source {path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 25
    try:
        while (ok_image := capture.read())[0]:
            yield ok_image[1]
            if not path.isdigit():
                time.sleep(1 / fps)  # play files at normal speed
    finally:
        capture.release()


def cmd_run(args: argparse.Namespace) -> None:
    last: list[tuple] = [()]

    def show(state: SignalState) -> None:
        key = (state.phase, state.light, state.mode)
        if key != last[0]:
            last[0] = key
            print(f"{state.phase.upper()} {state.light:8} mode={state.mode}")

    frames = video_file(args.source) if args.source else synthetic_camera(args.seconds, args.fps)
    engine = DetectionEngine(make_detector(args))
    pipeline = Pipeline(frames, engine, SignalController(load_config(args.config)), show)
    metrics = pipeline.run()
    print(json.dumps({k: round(v, 2) for k, v in metrics.summary().items()}, indent=2))
    if pipeline.camera_error:
        raise OSError(str(pipeline.camera_error))


def cmd_bench(args: argparse.Namespace) -> None:
    detector, counter, rng = make_detector(args), LaneCounter(), np.random.default_rng(99)
    times, errors, total = [], 0, 0
    for i in range(args.frames + 10):
        counts = {lane: int(rng.integers(0, VISIBLE + 1)) for lane in LANES}
        image, _ = render(counts, rng)
        started = time.perf_counter()
        found = counter.raw(detector.detect(image))
        if i >= 10:  # the first frames include model warm-up
            times.append((time.perf_counter() - started) * 1000)
            errors += sum(abs(found[x] - counts[x]) for x in LANES)
            total += sum(counts.values())
    p50, p95, p99 = np.percentile(times, [50, 95, 99])
    print(f"{args.detector} on {args.device}, {args.frames} frames at 640x640")
    print(f"latency p50 {p50:.1f} ms, p95 {p95:.1f} ms, p99 {p99:.1f} ms, max {max(times):.1f} ms")
    print(f"lane count error: {errors} of {total} vehicles ({errors / total:.2%})")


def cmd_evaluate(args: argparse.Namespace) -> None:
    table = markdown(compare(load_config(args.config), args.seeds))
    print(table)
    if args.output:
        args.output.write_text(table + "\n", encoding="utf-8")


def cmd_closed_loop(args: argparse.Namespace) -> None:
    config, detector = load_config(args.config), make_detector(args)
    scenario = SCENARIOS[args.scenario]
    for name, mode, sensor in (
        ("fixed", Mode.FIXED, camera_limit),
        ("adaptive, perfect counts", None, camera_limit),
        (f"adaptive, {args.detector} vision", None, vision_sensor(detector, args.seed)),
    ):
        j = simulate(scenario, config, mode, args.seed, sensor)
        print(f"{name:28} served {j.served:5}  mean delay {j.mean_delay_s():6.1f} s")


def cmd_train(args: argparse.Namespace) -> None:
    from .train import train

    metrics = train(args.dataset, args.weights, args.images, args.epochs, args.device)
    print(json.dumps({k: round(v, 4) for k, v in metrics.items()}, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="traffic-lights")
    parser.add_argument("--config", type=Path, default=CONFIG)
    sub = parser.add_subparsers(required=True)

    def add(name: str, func, help_text: str, detector: bool = False):
        p = sub.add_parser(name, help=help_text)
        p.set_defaults(func=func)
        p.add_argument("--weights", type=Path, default=Path("vehicles.pt"))
        p.add_argument("--device", default="cpu", help="cpu, mps or cuda")
        if detector:
            p.add_argument("--detector", choices=["yolo", "contour"], default="contour")
        return p

    p = add("run", cmd_run, "run the live pipeline", detector=True)
    p.add_argument("--source", help="video file, stream URL or camera number")
    p.add_argument("--seconds", type=float, default=20)
    p.add_argument("--fps", type=float, default=15)
    p = add("bench", cmd_bench, "measure latency and counting accuracy", detector=True)
    p.add_argument("--frames", type=int, default=200)
    p = add("evaluate", cmd_evaluate, "compare fixed and adaptive signals")
    p.add_argument("--seeds", type=int, default=20)
    p.add_argument("--output", type=Path)
    p = add("closed-loop", cmd_closed_loop, "simulate with vision in the loop", detector=True)
    p.add_argument("--scenario", choices=SCENARIOS, default="uneven")
    p.add_argument("--seed", type=int, default=0)
    p = add("train", cmd_train, "train YOLOv8n on synthetic frames")
    p.add_argument("--dataset", type=Path, default=Path("dataset"))
    p.add_argument("--images", type=int, default=300)
    p.add_argument("--epochs", type=int, default=25)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except (ConfigError, OSError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
