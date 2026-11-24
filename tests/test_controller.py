import pytest

from traffic_lights.config import ConfigError, JunctionConfig, Rule, load_config
from traffic_lights.controller import SignalController
from traffic_lights.messages import LANES, PHASES, LaneCounts, Light, Mode
from traffic_lights.rules import RuleEngine, queue_metrics

CONFIG = JunctionConfig()


def counts(ns=0, ew=0):
    return {lane: ns if lane in PHASES["ns"] else ew for lane in LANES}


def drive(controller, seconds, sense, step=0.5):
    """Returns the state at every tick; `sense(t)` gives lane counts or None for a dead camera."""
    states = []
    for i in range(int(seconds / step)):
        t = i * step
        c = sense(t)
        if c is not None:
            controller.observe(LaneCounts(i, t, c))
        states.append((t, controller.tick(t)))
    return states


def greens(states):
    """Completed green periods as (phase, length)."""
    out, start, previous = [], None, None
    for t, s in states:
        if s.light is Light.GREEN and (previous is None or previous.light is not Light.GREEN):
            start = t
        if s.light is Light.AMBER and previous.light is Light.GREEN:
            out.append((previous.phase, t - start))
        previous = s
    return out


def test_every_change_goes_green_amber_all_red():
    seen = []
    for _, s in drive(SignalController(CONFIG), 300, lambda t: counts(5, 5)):
        if not seen or seen[-1] != (s.phase, s.light):
            seen.append((s.phase, s.light))
    for a, b in zip(seen, seen[1:], strict=False):
        if a[1] is Light.GREEN:
            assert b == (a[0], Light.AMBER)
        elif a[1] is Light.AMBER:
            assert b == (a[0], Light.ALL_RED)
        else:
            assert b[1] is Light.GREEN and b[0] != a[0]


@pytest.mark.parametrize("mode", list(Mode))
@pytest.mark.parametrize("load", [(0, 1), (1, 1), (30, 1), (40, 40)])
def test_green_limits_hold_in_every_mode(mode, load):
    states = drive(SignalController(CONFIG, mode), 600, lambda t: counts(*load))
    periods = greens(states)
    assert periods
    assert all(CONFIG.min_green_s <= g <= CONFIG.max_green_s + 0.5 for _, g in periods)


def test_fixed_mode_is_fixed():
    periods = greens(drive(SignalController(CONFIG, Mode.FIXED), 400, lambda t: counts(30, 1)))
    assert {g for _, g in periods} == {30.0}


def test_peak_mode_shares_green_by_demand():
    periods = greens(drive(SignalController(CONFIG, Mode.PEAK), 400, lambda t: counts(12, 3)))
    ns = [g for p, g in periods if p == "ns"]
    ew = [g for p, g in periods if p == "ew"]
    assert min(ns[1:]) > max(ew)


def test_stays_green_when_nobody_else_waits():
    states = drive(SignalController(CONFIG, Mode.ADAPTIVE), 200, lambda t: counts(4, 0))
    assert {s.phase for _, s in states} == {"ns"}


def test_gap_out_ends_an_empty_green_early():
    periods = greens(
        drive(
            SignalController(CONFIG, Mode.ADAPTIVE),
            60,
            lambda t: counts(10 if t < 3 else 0, 5),
        )
    )
    assert periods[0] == ("ns", CONFIG.min_green_s)


def test_dead_camera_falls_back_to_fixed_time():
    states = drive(SignalController(CONFIG), 200, lambda t: counts(2, 9) if t < 10 else None)
    assert states[-1][1].mode is Mode.FIXED
    assert greens(states)[-1][1] == CONFIG.fixed_green_s


def test_out_of_order_detections_are_ignored():
    c = SignalController(CONFIG)
    c.observe(LaneCounts(2, 10.0, counts(5, 5)))
    c.observe(LaneCounts(1, 9.0, counts(0, 0)))
    assert c._load(0) == 20


def test_queue_metrics():
    m = queue_metrics(counts(3, 1), 0.5)
    assert (m["total_queue"], m["max_queue"], m["imbalance"]) == (16, 3, 0.5)
    assert queue_metrics({}, 0)["imbalance"] == 0


def test_rule_priority_hold_and_reset():
    busy = Rule("busy", "total_queue", ">=", 10, Mode.PEAK, priority=1, hold_s=5)
    stale = Rule("stale", "seconds_since_detection", ">", 3, Mode.FIXED, priority=9)
    engine = RuleEngine(JunctionConfig(rules=(busy, stale)))
    high, low = queue_metrics(counts(5, 5), 0), queue_metrics(counts(0, 0), 0)
    assert engine.mode(high, 0) is Mode.ADAPTIVE
    assert engine.mode(high, 5) is Mode.PEAK
    assert engine.mode(low, 6) is Mode.ADAPTIVE
    assert engine.mode(high, 7) is Mode.ADAPTIVE  # the hold timer restarted
    assert engine.mode(queue_metrics(counts(5, 5), 4), 20) is Mode.FIXED


@pytest.mark.parametrize(
    "text",
    [
        "min_green_s: 40",
        "amber_s: -1",
        "amber_s: yes",
        "nonsense: 1",
        "- a",
        "default_mode: turbo",
        "rules: [{name: a, metric: nope, op: '>', threshold: 1, mode: peak}]",
        "rules: [{name: a, metric: max_queue, op: '=', threshold: 1, mode: peak}]",
        "rules: [{name: a}]",
    ],
)
def test_bad_config_is_rejected(tmp_path, text):
    path = tmp_path / "c.yaml"
    path.write_text(text)
    with pytest.raises(ConfigError):
        load_config(path)


def test_shipped_config_loads():
    from pathlib import Path

    assert len(load_config(Path(__file__).parents[1] / "config/junction.yaml").rules) == 3
