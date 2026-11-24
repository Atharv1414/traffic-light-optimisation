# Architecture

| Stage | Input | Output |
|---|---|---|
| Camera ingestion | frames | `Frame` |
| Detection engine | `Frame` | `LaneCounts` |
| Signal controller | `LaneCounts` and time | `SignalState` |

Stages share only the typed messages in `messages.py`, so each could run as its own service. Here they are threads in one process.

## Decisions

- **The buffer holds one frame.** A slow detector skips frames rather than acting on old ones.
- **The controller takes time as an argument**, so simulation and live running use the same code.
- **Green lasts until the queue clears**, not for a time computed from the visible queue. The first version did the latter. The camera sees six vehicles per lane, so long queues got short greens, and evaluation showed it losing to fixed timing (throughput down 10.8% in the saturated-uneven case). The evaluation table caught it.
- **Safety limits are separate from the mode:** minimum green, maximum green while others wait, then amber and all-red.
- **Delay includes vehicles still queued** at the end, so leaving people waiting cannot look good.
- **A rolling median of three frames** removes single-frame miscounts.
- **YOLO is trained locally on synthetic frames**, so nothing is downloaded and results can be reproduced.

## Limits

- Synthetic camera only. Real deployment needs real labelled footage, camera calibration and night and weather testing.
- Two phases, no turns, pedestrians or emergency vehicles.
- Not certified for controlling real signals.
