"""Draw the junction as an overhead camera would see it, and know where every car is."""

import cv2
import numpy as np

from .messages import ARMS, LANES, Box

SIZE = 640
CENTRE = SIZE // 2
LANE_W = 34
STOP = 84  # distance from the centre to the stop line
CAR_L, CAR_W, GAP = 26, 16, 9
VISIBLE = (CENTRE - STOP) // (CAR_L + GAP)  # queue length the camera can see per lane


def _lane_offset(lane: str) -> int:
    return LANE_W // 2 + (int(lane[-1]) - 1) * LANE_W


def _place(arm: str, along: float, across: float) -> tuple[float, float]:
    """Distance from centre along an arm, and offset across it, to pixels (UK: drive on left)."""
    return {
        "north": (CENTRE + across, CENTRE - along),
        "south": (CENTRE - across, CENTRE + along),
        "east": (CENTRE + along, CENTRE + across),
        "west": (CENTRE - along, CENTRE - across),
    }[arm]


def lane_polygons() -> dict[str, np.ndarray]:
    polygons = {}
    for lane in LANES:
        arm, offset = lane.split("_")[0], _lane_offset(lane)
        near, far, half = STOP, CENTRE, LANE_W / 2
        corners = [(near, offset - half), (far, offset - half), (far, offset + half),
                   (near, offset + half)]  # fmt: skip
        polygons[lane] = np.array([_place(arm, *c) for c in corners], dtype=np.float32)
    return polygons


def render(counts: dict[str, int], rng: np.random.Generator) -> tuple[np.ndarray, list[Box]]:
    shade = int(rng.integers(60, 85))
    image = np.full((SIZE, SIZE, 3), (70, 110, 70), np.uint8)
    road = 2 * LANE_W
    cv2.rectangle(image, (CENTRE - road, 0), (CENTRE + road, SIZE), (shade,) * 3, -1)
    cv2.rectangle(image, (0, CENTRE - road), (SIZE, CENTRE + road), (shade,) * 3, -1)
    for arm in ARMS:
        a, b = _place(arm, STOP - 6, 0), _place(arm, STOP - 6, road)
        cv2.line(image, tuple(map(int, a)), tuple(map(int, b)), (235, 235, 235), 3)
    boxes = []
    for lane in LANES:
        arm = lane.split("_")[0]
        for slot in range(min(counts.get(lane, 0), VISIBLE)):
            along = STOP + CAR_L / 2 + slot * (CAR_L + GAP) + rng.uniform(-2, 2)
            x, y = _place(arm, along, _lane_offset(lane) + rng.uniform(-3, 3))
            w, h = (CAR_W, CAR_L) if arm in ("north", "south") else (CAR_L, CAR_W)
            p1 = (int(x - w / 2), int(y - h / 2))
            p2 = (p1[0] + w, p1[1] + h)
            colour = tuple(int(c) for c in rng.integers(0, 256, 3))
            cv2.rectangle(image, p1, p2, colour, -1)
            cv2.rectangle(image, p1, p2, (20, 20, 20), 1)
            inset = (p1[0] + w // 4, p1[1] + h // 4), (p2[0] - w // 4, p2[1] - h // 4)
            cv2.rectangle(image, *inset, (45, 40, 35), -1)
            boxes.append(Box(p1[0], p1[1], p2[0], p2[1]))
    noise = rng.normal(0, 6, image.shape)
    image = np.clip(image.astype(np.float32) * rng.uniform(0.75, 1.15) + noise, 0, 255)
    return image.astype(np.uint8), boxes
