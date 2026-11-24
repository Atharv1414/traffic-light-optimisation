"""Render docs/demo.gif: fixed and adaptive signals side by side on identical traffic.

The adaptive side runs the real loop: a frame is drawn, YOLOv8 detects the vehicles in it,
and the controller acts on those counts. Needs the `yolo` extra and trained weights.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from traffic_lights.config import load_config
from traffic_lights.controller import SignalController
from traffic_lights.detection import LaneCounter, YoloDetector
from traffic_lights.messages import ARMS, PHASES, LaneCounts, Light, Mode
from traffic_lights.scene import CENTRE, STOP, render
from traffic_lights.simulation import SCENARIOS, Junction

COLOURS = {Light.GREEN: (90, 220, 90), Light.AMBER: (60, 190, 255), Light.ALL_RED: (70, 70, 235)}
PANEL, HEADER = 360, 56


def draw_lights(image, state) -> None:
    for arm in ARMS:
        active = f"{arm}_1" in PHASES[state.phase]
        colour = COLOURS[state.light if active else Light.ALL_RED]
        d = STOP - 24
        x, y = {"north": (0, -d), "south": (0, d), "east": (d, 0), "west": (-d, 0)}[arm]
        cv2.circle(image, (CENTRE + x, CENTRE + y), 15, (25, 25, 25), -1)
        cv2.circle(image, (CENTRE + x, CENTRE + y), 11, colour, -1)


def panel(title: str, junction: Junction, state, image, boxes) -> np.ndarray:
    # Display only: even out brightness and noise, which would make the GIF flicker and grow.
    image = cv2.bilateralFilter(image, 7, 45, 5).astype(np.float32)
    grass = np.median(image[:150, :150], axis=(0, 1))
    image = np.clip(image * (np.array((70, 110, 70)) / grass), 0, 255).astype(np.uint8)
    for box in boxes:
        p1, p2 = (int(box.x1) - 2, int(box.y1) - 2), (int(box.x2) + 2, int(box.y2) + 2)
        cv2.rectangle(image, p1, p2, (255, 220, 60), 2)
    draw_lights(image, state)
    view = cv2.resize(image, (PANEL, PANEL), interpolation=cv2.INTER_AREA)
    head = np.full((HEADER, PANEL, 3), (32, 26, 22), np.uint8)
    font, white = cv2.FONT_HERSHEY_SIMPLEX, (245, 245, 245)
    cv2.putText(head, title, (12, 23), font, 0.56, white, 1, cv2.LINE_AA)
    stats = f"served {junction.served}   queued {sum(junction.counts().values())}"
    cv2.putText(head, stats, (12, 46), font, 0.5, (190, 190, 190), 1, cv2.LINE_AA)
    return np.vstack([head, view])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=Path("vehicles.pt"))
    parser.add_argument("--scenario", choices=SCENARIOS, default="saturated-uneven")
    parser.add_argument("--start", type=int, default=300, help="simulated seconds to skip")
    parser.add_argument("--seconds", type=int, default=150, help="simulated seconds to record")
    parser.add_argument("--every", type=int, default=2, help="simulated seconds per GIF frame")
    parser.add_argument("--output", type=Path, default=Path("docs/demo.gif"))
    args = parser.parse_args()

    config, detector = load_config(Path("config/junction.yaml")), YoloDetector(args.weights)
    names = ("Fixed timing", "Adaptive: YOLOv8 counts the queues")
    junctions = [Junction(SCENARIOS[args.scenario], config, seed=0) for _ in names]
    controllers = [SignalController(config, forced_mode=m) for m in (Mode.FIXED, None)]
    counter, frames, boxes = LaneCounter(), [], []

    for second in range(args.start + args.seconds):
        fixed, adaptive = junctions
        image, _ = render(adaptive.counts(), np.random.default_rng(7))
        boxes = detector.detect(image)
        controllers[0].observe(LaneCounts(second, fixed.now, fixed.counts()))
        controllers[1].observe(LaneCounts(second, adaptive.now, counter.update(boxes)))
        states = [c.tick(j.now) for c, j in zip(controllers, junctions, strict=True)]
        if second >= args.start and (second - args.start) % args.every == 0:
            other, _ = render(fixed.counts(), np.random.default_rng(7))
            left = panel(names[0], fixed, states[0], other, [])
            right = panel(names[1], adaptive, states[1], image, boxes)
            both = np.hstack([left, np.full((left.shape[0], 6, 3), 255, np.uint8), right])
            frames.append(Image.fromarray(cv2.cvtColor(both, cv2.COLOR_BGR2RGB)))
        for junction, controller in zip(junctions, controllers, strict=True):
            for _ in range(2):  # two half-second steps
                junction.step(controller.tick(junction.now))

    # Reserve palette space for the light colours, which cover too few pixels to survive alone.
    sample = frames[len(frames) // 2].copy()
    for i, (b, g, r) in enumerate([*COLOURS.values(), (255, 220, 60)]):
        sample.paste((r, g, b), (i * 160, 0, i * 160 + 160, 56))
    palette = sample.quantize(160, method=Image.Quantize.MEDIANCUT)
    frames = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
    frames[0].save(
        args.output, save_all=True, append_images=frames[1:], duration=160, loop=0, optimize=True
    )
    print(f"{args.output}: {len(frames)} frames, {args.output.stat().st_size / 1e6:.1f} MB")
    print(", ".join(f"{n}: {j.served} served" for n, j in zip(names, junctions, strict=True)))


if __name__ == "__main__":
    main()
