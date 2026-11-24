import math
from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml

from .messages import Mode

METRICS = {"total_queue", "max_queue", "imbalance", "seconds_since_detection"}
OPS = {">", ">=", "<", "<="}


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class Rule:
    name: str
    metric: str
    op: str
    threshold: float
    mode: Mode
    priority: int = 0
    hold_s: float = 0.0

    def __post_init__(self) -> None:
        if self.metric not in METRICS:
            raise ConfigError(f"rule {self.name}: unknown metric {self.metric!r}")
        if self.op not in OPS:
            raise ConfigError(f"rule {self.name}: unknown operator {self.op!r}")
        if self.hold_s < 0:
            raise ConfigError(f"rule {self.name}: hold_s must not be negative")


@dataclass(frozen=True)
class JunctionConfig:
    fixed_green_s: float = 30
    min_green_s: float = 7
    max_green_s: float = 55
    amber_s: float = 3
    all_red_s: float = 2
    saturation_flow: float = 0.5
    startup_lost_s: float = 2
    gap_out_s: float = 2
    stale_after_s: float = 3
    rules: tuple[Rule, ...] = field(default_factory=tuple)
    default_mode: Mode = Mode.ADAPTIVE

    def __post_init__(self) -> None:
        for f in fields(self):
            value = getattr(self, f.name)
            if f.name.endswith(("_s", "_flow")) and (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ConfigError(f"{f.name} must be a positive number")
        if not self.min_green_s <= self.fixed_green_s <= self.max_green_s:
            raise ConfigError("need min_green_s <= fixed_green_s <= max_green_s")
        if len({r.priority for r in self.rules}) != len(self.rules):
            raise ConfigError("rule priorities must be unique")


def load_config(path: Path) -> JunctionConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ConfigError("the config file must be a mapping")
    if unknown := set(raw) - {f.name for f in fields(JunctionConfig)}:
        raise ConfigError(f"unknown settings: {', '.join(sorted(unknown))}")
    try:
        rules = tuple(Rule(**{**r, "mode": Mode(r["mode"])}) for r in raw.pop("rules", []))
        if "default_mode" in raw:
            raw["default_mode"] = Mode(raw["default_mode"])
        return JunctionConfig(**raw, rules=rules)
    except ConfigError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise ConfigError(str(exc)) from exc
