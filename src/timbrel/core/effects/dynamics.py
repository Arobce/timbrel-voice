"""Noise gate and output limiter.

Both detect level once per block and apply a gain that ramps linearly across
the block, which keeps them vectorised and cheap enough to run all day.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

from timbrel.core.effects.base import Effect, db_to_gain
from timbrel.core.params import ParamSpec


class _GainRamp:
    """Multiplies a block by a gain ramping linearly from one value to another."""

    def __init__(self, max_block: int) -> None:
        self._frac = np.empty(max_block, dtype=np.float32)
        self._buf = np.empty(max_block, dtype=np.float32)
        self._n = 0

    def apply(self, block: np.ndarray, start: float, end: float) -> None:
        n = len(block)
        if start == end:
            if start != 1.0:
                block *= start
            return
        if n != self._n:
            self._n = n
            np.divide(np.arange(1, n + 1, dtype=np.float32), n, out=self._frac[:n])
        gain = self._buf[:n]
        np.multiply(self._frac[:n], end - start, out=gain)
        gain += start
        block *= gain


class NoiseGate(Effect):
    """Silences the mic between words.

    Opens when block RMS rises above the threshold, stays open for a short hold,
    then closes once the level drops 6 dB below the threshold (hysteresis stops
    it chattering on word endings), fading out over the release time.
    """

    name = "gate"
    PARAMS: Mapping[str, ParamSpec] = {
        "threshold_db": ParamSpec(-50.0, -70.0, -20.0, "dB"),
        "release_ms": ParamSpec(150.0, 20.0, 1000.0, "ms"),
    }
    HYSTERESIS_DB = 6.0
    HOLD_MS = 60.0
    ATTACK_MS = 1.0

    def __init__(self, sample_rate: int, block_size: int) -> None:
        super().__init__(sample_rate, block_size)
        self._ramp = _GainRamp(block_size)
        self._hold_samples = int(sample_rate * self.HOLD_MS / 1000)
        self._attack_samples = sample_rate * self.ATTACK_MS / 1000
        self.reset()

    def reset(self) -> None:
        self._gain = 0.0
        self._open = False
        self._hold_left = 0

    @property
    def is_open(self) -> bool:
        return self._open

    def _apply_params(self, params: Mapping[str, float]) -> None:
        self._open_level = db_to_gain(params["threshold_db"])
        self._close_level = db_to_gain(params["threshold_db"] - self.HYSTERESIS_DB)
        # Release = time to fade by 60 dB.
        self._release_samples = self.sample_rate * params["release_ms"] / 1000 / math.log(1000)

    def _process(self, block: np.ndarray) -> np.ndarray:
        n = len(block)
        rms = math.sqrt(float(np.dot(block, block)) / n) if n else 0.0

        if rms >= self._open_level:
            self._open = True
            self._hold_left = self._hold_samples
        elif self._open:
            if rms >= self._close_level:
                self._hold_left = self._hold_samples
            else:
                self._hold_left -= n
                if self._hold_left <= 0:
                    self._open = False

        start = self._gain
        if self._open:
            end = 1.0 - (1.0 - start) * math.exp(-n / self._attack_samples)
            if end > 0.999:
                end = 1.0
        else:
            end = start * math.exp(-n / self._release_samples)
            if end < 1e-4:
                end = 0.0
        self._gain = end
        self._ramp.apply(block, start, end)
        return block


class Limiter(Effect):
    """Keeps the output below the ceiling. Always last in the chain.

    Gain drops as soon as a block would exceed the ceiling and recovers over the
    release time. A final hard clip at the ceiling catches the start of a block
    while the gain is still ramping down.
    """

    name = "limiter"
    PARAMS: Mapping[str, ParamSpec] = {
        "ceiling_db": ParamSpec(-1.0, -12.0, 0.0, "dB"),
    }
    RELEASE_MS = 80.0

    def __init__(self, sample_rate: int, block_size: int) -> None:
        super().__init__(sample_rate, block_size)
        self._ramp = _GainRamp(block_size)
        self._release_samples = sample_rate * self.RELEASE_MS / 1000
        self.reset()

    def reset(self) -> None:
        self._gain = 1.0

    @property
    def gain_reduction_db(self) -> float:
        return float(-20.0 * np.log10(max(self._gain, 1e-10)))

    def _apply_params(self, params: Mapping[str, float]) -> None:
        self._ceiling = db_to_gain(params["ceiling_db"])

    def _process(self, block: np.ndarray) -> np.ndarray:
        n = len(block)
        if n == 0:
            return block
        peak = float(max(block.max(), -block.min()))
        start = self._gain
        released = 1.0 - (1.0 - start) * math.exp(-n / self._release_samples)
        if released > 0.9999:
            released = 1.0
        needed = self._ceiling / peak if peak > self._ceiling else 1.0
        end = min(released, needed)
        self._gain = end
        self._ramp.apply(block, start, end)
        if peak * max(start, end) > self._ceiling:
            np.clip(block, -self._ceiling, self._ceiling, out=block)
        return block
