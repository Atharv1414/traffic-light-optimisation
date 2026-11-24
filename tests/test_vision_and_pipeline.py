import threading
import time

import numpy as np
import pytest

from traffic_lights.cli import main
from traffic_lights.config import JunctionConfig
from traffic_lights.controller import SignalController
from traffic_lights.detection import ContourDetector, DetectionEngine, LaneCounter, YoloDetector
from traffic_lights.evaluate import camera_limit, compare, markdown, simulate, vision_sensor
from traffic_lights.messages import LANES, Box, Frame, Mode
from traffic_lights.pipeline import LatestFrameBuffer, Pipeline
from traffic_lights.scene import VISIBLE, render
from traffic_lights.simulation import SCENARIOS, Scenario
from traffic_lights.train import write_dataset

CONFIG = JunctionConfig()


def random_counts(rng):
    return {lane: int(rng.integers(0, VISIBLE + 1)) for lane in LANES}


def test_true_boxes_map_to_the_right_lanes():
    rng = np.random.default_rng(1)
    for _ in range(20):
        counts = random_counts(rng)
        _, boxes = render(counts, rng)
        assert LaneCounter().raw(boxes) == counts


def test_boxes_outside_lanes_are_ignored():
    assert sum(LaneCounter().raw([Box(0, 0, 20, 20), Box(310, 310, 330, 330)]).values()) == 0


def test_contour_detector_is_accurate_on_the_synthetic_camera():
    rng = np.random.default_rng(2)
    errors = total = 0
    for _ in range(30):
        counts = random_counts(rng)
        found = LaneCounter().raw(ContourDetector().detect(render(counts, rng)[0]))
        errors += sum(abs(found[x] - counts[x]) for x in LANES)
        total += sum(counts.values())
    assert errors / total < 0.1


def test_counter_median_removes_a_one_frame_glitch():
    counter = LaneCounter(window=3)
    polygon_box = render({"north_1": 1}, np.random.default_rng(0))[1]
    for boxes in (polygon_box, [], polygon_box):
        counts = counter.update(boxes)
    assert counts["north_1"] == 1


def test_missing_weights_have_a_clear_error(tmp_path):
    pytest.importorskip("ultralytics")
    with pytest.raises(FileNotFoundError, match="train"):
        YoloDetector(tmp_path / "none.pt")


def test_dataset_labels_are_normalised(tmp_path):
    write_dataset(tmp_path, 3, 2)
    assert len(list((tmp_path / "images/train").iterdir())) == 3
    for path in (tmp_path / "labels").rglob("*.txt"):
        for line in path.read_text().splitlines():
            values = [float(v) for v in line.split()]
            assert values[0] == 0 and all(0 <= v <= 1 for v in values[1:])


def frame(n):
    return Frame(n, time.monotonic(), np.zeros((2, 2, 3), np.uint8))


def test_buffer_keeps_only_the_newest_frame():
    buffer = LatestFrameBuffer()
    for n in range(5):
        buffer.put(frame(n))
    assert buffer.get().sequence == 4
    assert buffer.dropped == 4
    assert buffer.get(timeout=0.01) is None


def test_closing_the_buffer_wakes_a_waiting_reader():
    buffer, done = LatestFrameBuffer(), threading.Event()
    reader = threading.Thread(target=lambda: (buffer.get(), done.set()), daemon=True)
    reader.start()
    buffer.close()
    assert done.wait(1)


class Fake:
    def __init__(self, delay=0.0, fail_on=()):
        self.delay, self.fail_on, self.calls = delay, set(fail_on), 0

    def detect(self, image):
        self.calls += 1
        time.sleep(self.delay)
        if self.calls in self.fail_on:
            raise RuntimeError("bad frame")
        return []


def camera(n, gap):
    for _ in range(n):
        yield np.zeros((8, 8, 3), np.uint8)
        time.sleep(gap)


def run(detector, frames):
    return Pipeline(frames, DetectionEngine(detector), SignalController(CONFIG)).run()


def test_pipeline_accounts_for_every_frame():
    m = run(Fake(delay=0.03), camera(30, 0.005))
    assert m.captured == 30
    assert m.dropped > 0
    assert m.processed + m.dropped == m.captured


def test_slow_detector_does_not_build_up_lag():
    m = run(Fake(delay=0.04), camera(60, 0.005))
    assert max(m.end_to_end_ms) < 400  # a queue of 60 frames would reach about 2,400 ms


def test_pipeline_survives_detector_errors():
    m = run(Fake(fail_on={1}), camera(10, 0.02))
    assert m.detector_errors == 1
    assert m.processed > 0


def test_pipeline_survives_a_camera_crash():
    def broken():
        yield np.zeros((8, 8, 3), np.uint8)
        time.sleep(0.05)
        raise OSError("unplugged")

    assert run(Fake(), broken()).processed == 1


def test_vehicles_are_conserved():
    for mode in (Mode.FIXED, None):
        j = simulate(SCENARIOS["tidal"], CONFIG, mode, 3)
        assert j.arrived == j.served + sum(j.counts().values())


def test_same_seed_gives_identical_arrivals_to_both_controllers():
    a = simulate(SCENARIOS["balanced"], CONFIG, Mode.FIXED, 5)
    b = simulate(SCENARIOS["balanced"], CONFIG, None, 5)
    assert a.arrived == b.arrived


def test_flow_never_beats_physical_capacity():
    j = simulate(SCENARIOS["saturated-balanced"], CONFIG, None, 0)
    assert j.served <= 4 * CONFIG.saturation_flow * 3600


def test_adaptive_cuts_delay_when_traffic_is_uneven():
    fixed = simulate(SCENARIOS["uneven"], CONFIG, Mode.FIXED, 0)
    adaptive = simulate(SCENARIOS["uneven"], CONFIG, None, 0)
    assert adaptive.mean_delay_s() < fixed.mean_delay_s()


def test_vision_in_the_loop_is_close_to_perfect_counts():
    short = Scenario("short", (700, 700), (300, 300), duration_s=300)
    ideal = simulate(short, CONFIG, None, 0, camera_limit)
    vision = simulate(short, CONFIG, None, 0, vision_sensor(ContourDetector(), 0))
    assert abs(vision.served - ideal.served) / ideal.served < 0.1


def test_comparison_table():
    results = compare(CONFIG, seeds=2)
    assert len(markdown(results).splitlines()) == len(SCENARIOS) + 2


def test_cli(capsys):
    assert main(["bench", "--frames", "5"]) == 0
    assert "latency" in capsys.readouterr().out
    assert main(["run", "--seconds", "1", "--fps", "10"]) == 0
    assert main(["run", "--source", "missing.mp4"]) == 1
