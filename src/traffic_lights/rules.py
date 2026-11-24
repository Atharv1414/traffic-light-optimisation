"""Threshold rules that choose the operating mode, with a hold time to stop flapping."""

import operator
from collections.abc import Mapping

from .config import JunctionConfig, Rule
from .messages import PHASES, Mode

OPERATORS = {">": operator.gt, ">=": operator.ge, "<": operator.lt, "<=": operator.le}


def queue_metrics(counts: Mapping[str, int], seconds_since_detection: float) -> dict[str, float]:
    loads = [sum(counts.get(lane, 0) for lane in lanes) for lanes in PHASES.values()]
    total = sum(loads)
    return {
        "total_queue": total,
        "max_queue": max(counts.values(), default=0),
        "imbalance": (max(loads) - min(loads)) / total if total else 0.0,
        "seconds_since_detection": seconds_since_detection,
    }


class RuleEngine:
    def __init__(self, config: JunctionConfig) -> None:
        self._rules = sorted(config.rules, key=lambda r: -r.priority)
        self._default = config.default_mode
        self._matched_since: dict[str, float] = {}

    def mode(self, metrics: Mapping[str, float], now: float) -> Mode:
        chosen: Rule | None = None
        for rule in self._rules:
            if OPERATORS[rule.op](metrics[rule.metric], rule.threshold):
                since = self._matched_since.setdefault(rule.name, now)
                if chosen is None and now - since >= rule.hold_s:
                    chosen = rule
            else:
                self._matched_since.pop(rule.name, None)
        return chosen.mode if chosen else self._default
