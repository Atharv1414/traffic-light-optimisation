"""Threaded runtime: camera -> latest-frame buffer -> detection -> controller."""

import logging
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field

import numpy as np

from .controller import SignalController
from .detection import DetectionEngine
from .messages import Frame, SignalState

log = logging.getLogger(__name__)


class LatestFrameBuffer:
    """Holds one frame. A slow detector skips old frames instead of building up lag."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._frame: Frame | None = None
        self._closed = False
        self.dropped = 0

    def put(self, frame: Frame) -> None:
        with self._condition:
            if self._frame is not None:
                self.dropped += 1
            self._frame = frame
            self._condition.notify()

    def get(self, timeout: float | None = None) -> Frame | None:
        """The newest unseen frame, or None on timeout or once closed and empty."""
        with self._condition:
            self._condition.wait_for(lambda: self._frame is not None or self._closed, timeout)
            frame, self._frame = self._frame, None
            return frame

    @property
    def closed(self) -> bool:
        with self._condition:
            return self._closed and self._frame is None

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()


@dataclass
class Metrics:
    captured: int = 0
    processed: int = 0
    dropped: int = 0
    detector_errors: int = 0
    inference_ms: list[float] = field(default_factory=list)
    end_to_end_ms: list[float] = field(default_factory=list)

    def summary(self) -> dict[str, float]:
        out: dict[str, float] = {
            "captured": self.captured,
            "processed": self.processed,
            "dropped": self.dropped,
            "detector_errors": self.detector_errors,
        }
        for name, values in (("inference", self.inference_ms), ("end_to_end", self.end_to_end_ms)):
            if values:
                for p in (50, 95, 99):
                    out[f"{name}_p{p}_ms"] = float(np.percentile(values, p))
        return out


class Pipeline:
    def __init__(
        self,
        frames: Iterator[np.ndarray],
        engine: DetectionEngine,
        controller: SignalController,
        on_state: Callable[[SignalState], None] = lambda s: None,
        tick_s: float = 0.05,
    ) -> None:
        self._frames, self._engine, self._controller = frames, engine, controller
        self._on_state, self._tick_s = on_state, tick_s
        self._buffer = LatestFrameBuffer()
        self._lock = threading.Lock()  # the controller is shared by two threads
        self._stop = threading.Event()
        self.camera_error: Exception | None = None
        self.metrics = Metrics()

    def stop(self) -> None:
        self._stop.set()

    def _ingest(self) -> None:
        try:
            for sequence, image in enumerate(self._frames):
                if self._stop.is_set():
                    break
                self._buffer.put(Frame(sequence, time.monotonic(), image))
                self.metrics.captured += 1
        except Exception as exc:
            self.camera_error = exc
            log.error("camera failed: %s", exc)
        finally:
            self._buffer.close()

    def _detect(self) -> None:
        while not self._stop.is_set() and not self._buffer.closed:
            frame = self._buffer.get(timeout=0.2)
            if frame is None:
                continue
            started = time.perf_counter()
            try:
                counts = self._engine.process(frame)
            except Exception:  # one bad frame must not stop the signals
                self.metrics.detector_errors += 1
                log.exception("detection failed on frame %d", frame.sequence)
                continue
            self.metrics.inference_ms.append((time.perf_counter() - started) * 1000)
            with self._lock:
                self._controller.observe(counts)
            self.metrics.end_to_end_ms.append((time.monotonic() - frame.captured_at) * 1000)
            self.metrics.processed += 1

    def run(self) -> Metrics:
        threads = [
            threading.Thread(target=self._ingest, name="camera", daemon=True),
            threading.Thread(target=self._detect, name="detector", daemon=True),
        ]
        for t in threads:
            t.start()
        try:
            while threads[1].is_alive():
                with self._lock:
                    state = self._controller.tick(time.monotonic())
                self._on_state(state)
                self._stop.wait(self._tick_s)
        finally:
            self.stop()
            for t in threads:
                t.join(timeout=5)
        self.metrics.dropped = self._buffer.dropped
        return self.metrics
