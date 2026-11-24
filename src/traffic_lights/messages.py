"""Typed messages. Pipeline stages know these and nothing else about each other."""

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

ARMS = ("north", "south", "east", "west")
LANES = tuple(f"{arm}_{n}" for arm in ARMS for n in (1, 2))
PHASES: dict[str, tuple[str, ...]] = {
    "ns": ("north_1", "north_2", "south_1", "south_2"),
    "ew": ("east_1", "east_2", "west_1", "west_2"),
}


class Mode(StrEnum):
    FIXED = "fixed"
    ADAPTIVE = "adaptive"
    PEAK = "peak"
    QUIET = "quiet"


class Light(StrEnum):
    GREEN = "green"
    AMBER = "amber"
    ALL_RED = "all_red"


@dataclass(frozen=True)
class Frame:
    sequence: int
    captured_at: float
    image: np.ndarray


@dataclass(frozen=True)
class Box:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float = 1.0

    @property
    def centre(self) -> tuple[float, float]:
        return (self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2


@dataclass(frozen=True)
class LaneCounts:
    sequence: int
    captured_at: float
    counts: dict[str, int]


@dataclass(frozen=True)
class SignalState:
    phase: str
    light: Light
    mode: Mode
    elapsed_s: float
