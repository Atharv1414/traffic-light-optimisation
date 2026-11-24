"""Compare controllers on identical traffic. Nothing here is tuned to reach a target number."""

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .config import JunctionConfig
from .controller import SignalController
from .detection import Detector, LaneCounter
from .messages import LaneCounts, Mode
from .scene import VISIBLE, render
from .simulation import SCENARIOS, Junction, Scenario

Sensor = Callable[[dict[str, int]], dict[str, int]]


def camera_limit(counts: dict[str, int]) -> dict[str, int]:
    """Perfect detection, but only as far back as the camera can see."""
    return {lane: min(n, VISIBLE) for lane, n in counts.items()}


def vision_sensor(detector: Detector, seed: int) -> Sensor:
    """Closed loop: draw a frame of the queues, detect vehicles in it, count per lane."""
    rng = np.random.default_rng(seed)
    counter = LaneCounter()

    def sense(counts: dict[str, int]) -> dict[str, int]:
        image, _ = render(counts, rng)
        return counter.update(detector.detect(image))

    return sense


def simulate(
    scenario: Scenario,
    config: JunctionConfig,
    mode: Mode | None,
    seed: int,
    sensor: Sensor = camera_limit,
) -> Junction:
    junction = Junction(scenario, config, seed)
    controller = SignalController(config, forced_mode=mode)
    sequence = 0
    while junction.now < scenario.duration_s:
        if junction.now == int(junction.now):  # one detection per simulated second
            counts = sensor(junction.counts())
            controller.observe(LaneCounts(sequence, junction.now, counts))
            sequence += 1
        junction.step(controller.tick(junction.now))
    return junction


@dataclass(frozen=True)
class Result:
    scenario: str
    fixed_vph: float
    adaptive_vph: float
    throughput_change: float
    throughput_ci: float  # 95% half-width
    fixed_delay_s: float
    adaptive_delay_s: float
    delay_change: float
    fixed_max_wait_s: float
    adaptive_max_wait_s: float


def compare(config: JunctionConfig, seeds: int = 20) -> list[Result]:
    results = []
    for scenario in SCENARIOS.values():
        fixed = [simulate(scenario, config, Mode.FIXED, s) for s in range(seeds)]
        adaptive = [simulate(scenario, config, None, s) for s in range(seeds)]
        hours = scenario.duration_s / 3600
        change = np.array([a.served / f.served - 1 for a, f in zip(adaptive, fixed, strict=True)])
        spread = change.std(ddof=1) if seeds > 1 else 0.0
        f_delay = float(np.mean([j.mean_delay_s() for j in fixed]))
        a_delay = float(np.mean([j.mean_delay_s() for j in adaptive]))
        results.append(
            Result(
                scenario.name,
                float(np.mean([j.served for j in fixed])) / hours,
                float(np.mean([j.served for j in adaptive])) / hours,
                float(change.mean()),
                float(1.96 * spread / math.sqrt(seeds)),
                f_delay,
                a_delay,
                a_delay / f_delay - 1,
                float(np.mean([j.max_wait_s for j in fixed])),
                float(np.mean([j.max_wait_s for j in adaptive])),
            )
        )
    return results


def markdown(results: list[Result]) -> str:
    lines = [
        "| Scenario | Fixed veh/h | Adaptive veh/h | Throughput (95% CI) "
        "| Mean delay, fixed | Mean delay, adaptive | Delay change |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    lines += [
        f"| {r.scenario} | {r.fixed_vph:.0f} | {r.adaptive_vph:.0f} "
        f"| {r.throughput_change:+.1%} ± {r.throughput_ci:.1%} "
        f"| {r.fixed_delay_s:.1f} s | {r.adaptive_delay_s:.1f} s | {r.delay_change:+.0%} |"
        for r in results
    ]
    return "\n".join(lines)
