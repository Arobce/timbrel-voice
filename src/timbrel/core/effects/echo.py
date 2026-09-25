"""Echo: a feedback delay line with damping, for rooms, caves and halls."""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
from scipy import signal

from timbrel.core.effects.base import Effect
from timbrel.core.params import ParamSpec, SmoothedValue

BUFFER_SIZE = 65536  # power of two; > max time (1.2 s at 48 kHz) + a block
DAMPING_HZ = 3500.0  # each repeat loses some treble, like a real room


class Echo(Effect):
    """out = dry + mix * delayed; delayed repeats are fed back and damped.

    The minimum time (30 ms) is longer than the largest block (1024 samples at
    48 kHz), so each block's echo only reads audio written in earlier blocks
    and the whole block can be processed at once.
    """

    name = "echo"
    PARAMS: Mapping[str, ParamSpec] = {
        "time_ms": ParamSpec(250.0, 30.0, 1200.0, "ms"),
        "feedback": ParamSpec(0.35, 0.0, 0.9),
        "mix": ParamSpec(0.3, 0.0, 1.0),
    }

    def __init__(self, sample_rate: int, block_size: int) -> None:
        super().__init__(sample_rate, block_size)
        min_delay = sample_rate * self.PARAMS["time_ms"].min / 1000
        if block_size >= min_delay:
            raise ValueError("block size must be shorter than the minimum echo time")
        self._mask = BUFFER_SIZE - 1
        self._buf = np.zeros(BUFFER_SIZE, dtype=np.float64)
        # Delay changes glide over ~50 ms (a short tape-style pitch bend, no clicks).
        glide = int(sample_rate * 0.05)
        self._delay = SmoothedValue(0.0, glide, block_size)
        self._feedback = SmoothedValue(0.0, self.ramp_samples, block_size)
        self._mix = SmoothedValue(0.0, self.ramp_samples, block_size)
        self._first = True
        coeff = math.exp(-2 * math.pi * DAMPING_HZ / sample_rate)
        self._damp_b = np.array([1.0 - coeff])
        self._damp_a = np.array([1.0, -coeff])
        self._damp_zi = np.zeros(1)
        self._offsets = np.arange(block_size, dtype=np.float64)
        self._pos = np.empty(block_size, dtype=np.float64)
        self._frac = np.empty(block_size, dtype=np.float64)
        self._idx = np.empty(block_size, dtype=np.int64)
        self._wet = np.empty(block_size, dtype=np.float64)
        self._wet1 = np.empty(block_size, dtype=np.float64)
        self._feed = np.empty(block_size, dtype=np.float64)
        self._write = 0

    def reset(self) -> None:
        self._buf.fill(0.0)
        self._damp_zi.fill(0.0)
        self._write = 0

    def _apply_params(self, params: Mapping[str, float]) -> None:
        delay = self.sample_rate * params["time_ms"] / 1000
        if self._first:
            self._first = False
            self._delay.jump(delay)
            self._feedback.jump(params["feedback"])
            self._mix.jump(params["mix"])
            return
        self._delay.set_target(delay)
        self._feedback.set_target(params["feedback"])
        self._mix.set_target(params["mix"])

    def _process(self, block: np.ndarray) -> np.ndarray:
        n = len(block)
        # Read the echo: position = write index - delay, linearly interpolated.
        pos, frac, idx = self._pos[:n], self._frac[:n], self._idx[:n]
        np.subtract(self._offsets[:n], self._delay.next_block(n), out=pos)
        pos += self._write + BUFFER_SIZE
        np.floor(pos, out=frac)
        np.copyto(idx, frac, casting="unsafe")
        np.subtract(pos, frac, out=frac)
        np.bitwise_and(idx, self._mask, out=idx)
        wet, wet1 = self._wet[:n], self._wet1[:n]
        np.take(self._buf, idx, out=wet)
        idx += 1
        np.bitwise_and(idx, self._mask, out=idx)
        np.take(self._buf, idx, out=wet1)
        wet1 -= wet
        wet1 *= frac
        wet += wet1

        damped, self._damp_zi = signal.lfilter(self._damp_b, self._damp_a, wet, zi=self._damp_zi)

        # Write input + damped feedback into the line.
        feed = self._feed[:n]
        np.multiply(damped, self._feedback.next_block(n), out=feed)
        feed += block
        start = self._write & self._mask
        first = min(n, BUFFER_SIZE - start)
        self._buf[start : start + first] = feed[:first]
        if first < n:
            self._buf[: n - first] = feed[first:]
        self._write += n

        wet *= self._mix.next_block(n)
        np.add(block, wet, out=block, casting="same_kind")
        return block
