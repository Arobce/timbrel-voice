"""Measure real round-trip latency through a full-duplex stream.

Plays clicks into an output device and records them back from an input device
using the same stream settings as the engine. With VB-Cable, the loop is
CABLE Input -> CABLE Output, so the measured time is the delay the engine's
stream adds (input buffering + output buffering), excluding effect latency.

    python scripts/latency_test.py
    python scripts/latency_test.py --block-size 128 --clicks 10
"""

from __future__ import annotations

import argparse
import statistics
import sys
import threading
from typing import Any

import numpy as np
import sounddevice as sd

from timbrel.core.engine import DEFAULT_BLOCK_SIZE, DEFAULT_SAMPLE_RATE
from timbrel.platform import windows

CLICK_INTERVAL_S = 0.5
CLICK_SAMPLES = 48  # 1 ms


def measure(
    input_device: int,
    output_device: int,
    block_size: int,
    clicks: int,
    exclusive_capture: bool = False,
) -> tuple[list[float], tuple[float, float]]:
    sr = DEFAULT_SAMPLE_RATE
    interval = int(CLICK_INTERVAL_S * sr)
    warmup = interval  # first click after 0.5 s
    total = warmup + interval * clicks + sr // 2
    recording = np.zeros(total, dtype=np.float32)
    click_starts = [warmup + i * interval for i in range(clicks)]
    signal = np.zeros(total, dtype=np.float32)
    for start in click_starts:
        signal[start : start + CLICK_SAMPLES] = 0.8
    pos = 0
    done = threading.Event()

    def callback(
        indata: np.ndarray, outdata: np.ndarray, frames: int, time: Any, status: Any
    ) -> None:
        nonlocal pos
        n = min(frames, total - pos)
        outdata[:n] = signal[pos : pos + n, np.newaxis]
        outdata[n:] = 0
        recording[pos : pos + n] = indata[:n, 0]
        pos += n
        if pos >= total:
            done.set()

    wasapi = windows.query_wasapi_devices()
    out_dev = wasapi.by_index(output_device)
    stream = sd.Stream(
        device=(input_device, output_device),
        samplerate=sr,
        blocksize=block_size,
        dtype="float32",
        channels=(1, 2),
        latency="low",
        extra_settings=(
            # Capture stands in for the mic side of the engine.
            windows.stream_settings(exclusive=exclusive_capture),
            windows.stream_settings(out_dev),  # same mode the engine uses
        ),
        callback=callback,
    )
    with stream:
        done.wait(timeout=total / sr + 5)
        reported = stream.latency

    threshold = 0.2
    delays = []
    for start in click_starts:
        window = recording[start : start + interval]
        hits = np.flatnonzero(np.abs(window) > threshold)
        if hits.size:
            delays.append(hits[0] / sr * 1000.0)
    return delays, reported


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", default="CABLE Output", help="loopback capture device")
    parser.add_argument("--output", help="loopback playback device (default: VB-Cable)")
    parser.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE)
    parser.add_argument("--clicks", type=int, default=8)
    parser.add_argument(
        "--exclusive-capture",
        action="store_true",
        help="capture in exclusive mode, like timbrel --exclusive-mic",
    )
    args = parser.parse_args()

    wasapi = windows.query_wasapi_devices()
    try:
        mic = windows.resolve_device(args.input, "input", wasapi)
        out = windows.resolve_device(args.output, "output", wasapi)
    except windows.DeviceError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"Loopback: [{out.index}] {out.name} -> [{mic.index}] {mic.name}")
    delays, reported = measure(
        mic.index, out.index, args.block_size, args.clicks, args.exclusive_capture
    )
    print(
        f"Block {args.block_size}: PortAudio reports {reported[0] * 1000:.1f} ms in + "
        f"{reported[1] * 1000:.1f} ms out = {sum(reported) * 1000:.1f} ms"
    )
    if not delays:
        print("No clicks detected. Is the loopback routed correctly?", file=sys.stderr)
        return 1
    print(
        f"Measured round trip over {len(delays)}/{args.clicks} clicks: "
        f"median {statistics.median(delays):.1f} ms, "
        f"min {min(delays):.1f} ms, max {max(delays):.1f} ms"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
