"""Queue simulation of the junction, used for evaluation and as the camera in demos."""

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .config import JunctionConfig
from .messages import LANES, PHASES, Light, SignalState

STEP_S = 0.5


@dataclass(frozen=True)
class Scenario:
    """Arrival rates in vehicles per hour per lane, at the start and end of the run."""

    name: str
    ns_vph: tuple[float, float]
    ew_vph: tuple[float, float]
    duration_s: int = 3600

    def rate(self, lane: str, t: float) -> float:
        start, end = self.ns_vph if lane in PHASES["ns"] else self.ew_vph
        return (start + (end - start) * t / self.duration_s) / 3600


SCENARIOS = {
    s.name: s
    for s in (
        Scenario("quiet", (90, 90), (90, 90)),
        Scenario("balanced", (500, 500), (500, 500)),
        Scenario("uneven", (850, 850), (250, 250)),
        Scenario("tidal", (1000, 250), (250, 1000)),
        Scenario("saturated-uneven", (1300, 1300), (350, 350)),
        Scenario("saturated-balanced", (950, 950), (950, 950)),
    )
}


@dataclass
class Junction:
    scenario: Scenario
    config: JunctionConfig
    seed: int = 0
    now: float = 0.0
    queues: dict[str, deque[float]] = field(default_factory=lambda: {x: deque() for x in LANES})
    served: int = 0
    arrived: int = 0
    total_delay_s: float = 0.0
    max_wait_s: float = 0.0

    def __post_init__(self) -> None:
        # Arrivals come from their own generator, so every controller sees the same traffic.
        self._rng = np.random.default_rng(self.seed)
        self._credit = dict.fromkeys(LANES, 0.0)

    def counts(self) -> dict[str, int]:
        return {lane: len(q) for lane, q in self.queues.items()}

    def step(self, state: SignalState) -> None:
        rates = np.array([self.scenario.rate(lane, self.now) for lane in LANES])
        for lane, n in zip(LANES, self._rng.poisson(rates * STEP_S), strict=True):
            self.queues[lane].extend([self.now] * int(n))
            self.arrived += int(n)
        for lane, queue in self.queues.items():
            flowing = (
                state.light is Light.GREEN
                and lane in PHASES[state.phase]
                and state.elapsed_s >= self.config.startup_lost_s
            )
            if not flowing:
                self._credit[lane] = 0.0
                continue
            self._credit[lane] += self.config.saturation_flow * STEP_S
            while queue and self._credit[lane] >= 1:
                wait = self.now - queue.popleft()
                self.served += 1
                self.total_delay_s += wait
                self.max_wait_s = max(self.max_wait_s, wait)
                self._credit[lane] -= 1
            if not queue:
                self._credit[lane] = min(self._credit[lane], 1.0)  # no banking unused green
        self.now += STEP_S

    def mean_delay_s(self) -> float:
        """Includes vehicles still queued, so leaving people waiting cannot look good."""
        waiting = sum(self.now - t for q in self.queues.values() for t in q)
        return (self.total_delay_s + waiting) / self.arrived if self.arrived else 0.0
