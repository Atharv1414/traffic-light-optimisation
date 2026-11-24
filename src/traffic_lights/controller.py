"""Signal controller: a safe state machine whose green length depends on measured queues."""

from .config import JunctionConfig
from .messages import PHASES, LaneCounts, Light, Mode, SignalState
from .rules import RuleEngine, queue_metrics

ORDER = tuple(PHASES)
CLEARED = 1  # a lane with at most this many vehicles has no queue left, only new arrivals


class SignalController:
    """Call `tick` regularly. Safety limits hold in every mode:

    green is never shorter than `min_green_s` or longer than `max_green_s` when another phase
    has traffic waiting, and every change passes through amber and all-red.
    """

    def __init__(self, config: JunctionConfig, forced_mode: Mode | None = None) -> None:
        self.config = config
        self._forced = forced_mode
        self._rules = RuleEngine(config)
        self._phase = 0
        self._light = Light.GREEN
        self._light_since: float | None = None
        self._cleared_since: float | None = None
        self._counts: dict[str, int] = {}
        self._last_detection: float | None = None
        self._mode = forced_mode or config.default_mode

    def observe(self, counts: LaneCounts) -> None:
        if self._last_detection is not None and counts.captured_at < self._last_detection:
            return  # out-of-order result from a slow frame
        self._counts = counts.counts
        self._last_detection = counts.captured_at

    def _load(self, phase: int) -> int:
        return sum(self._counts.get(lane, 0) for lane in PHASES[ORDER[phase]])

    def _longest_lane(self, phase: int) -> int:
        return max(self._counts.get(lane, 0) for lane in PHASES[ORDER[phase]])

    def _green_cap(self) -> float:
        """Longest green allowed now. Peak mode shares long greens out by measured demand."""
        c = self.config
        if self._mode is Mode.PEAK:
            total = sum(self._counts.values())
            share = self._load(self._phase) / total if total else 0.5
            return min(max(2 * c.max_green_s * share, c.min_green_s), c.max_green_s)
        return c.max_green_s

    def _should_end_green(self, elapsed: float, now: float) -> bool:
        c = self.config
        if elapsed < c.min_green_s:
            return False
        if self._mode is Mode.FIXED:
            return elapsed >= c.fixed_green_s
        if not self._load((self._phase + 1) % len(ORDER)):
            return False  # nobody is waiting, so stay green
        if self._mode is Mode.QUIET and self._cleared_since is not None:
            return True
        # The camera sees only the front of a long queue, so a full view says nothing about how
        # much green is needed. Hold green while a queue is still discharging instead.
        cleared = self._cleared_since is not None and now - self._cleared_since >= c.gap_out_s
        return cleared or elapsed >= self._green_cap()

    def tick(self, now: float) -> SignalState:
        if self._light_since is None:
            self._light_since = now
        if self._last_detection is None:
            self._last_detection = now
        age = now - self._last_detection
        if self._forced is None:
            self._mode = self._rules.mode(queue_metrics(self._counts, age), now)
            if age > self.config.stale_after_s:
                self._mode = Mode.FIXED  # built in, so a bad rules file cannot remove it
        self._step(now)
        return SignalState(ORDER[self._phase], self._light, self._mode, now - self._light_since)

    def _step(self, now: float) -> None:
        assert self._light_since is not None
        elapsed = now - self._light_since
        if self._light is Light.GREEN:
            if self._longest_lane(self._phase) > CLEARED:
                self._cleared_since = None
            elif self._cleared_since is None:
                self._cleared_since = now
            if self._should_end_green(elapsed, now):
                self._change(Light.AMBER, now)
        elif self._light is Light.AMBER and elapsed >= self.config.amber_s:
            self._change(Light.ALL_RED, now)
        elif self._light is Light.ALL_RED and elapsed >= self.config.all_red_s:
            self._phase = (self._phase + 1) % len(ORDER)
            self._cleared_since = None
            self._change(Light.GREEN, now)

    def _change(self, light: Light, now: float) -> None:
        self._light = light
        self._light_since = now
