"""Radio / walkie-talkie: static, telephone band-pass and soft clipping."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from scipy import signal

from timbrel.core.effects.base import Effect
from timbrel.core.params import ParamSpec, SmoothedValue

LOW_HZ = 300.0
HIGH_HZ = 3400.0
MAX_DRIVE_GAIN = 20.0
MAX_NOISE_AMPLITUDE = 0.03


class Radio(Effect):
    name = "radio"
    PARAMS: Mapping[str, ParamSpec] = {
        "drive": ParamSpec(0.5, 0.0, 1.0),
        "noise": ParamSpec(0.2, 0.0, 1.0),
    }

    def __init__(self, sample_rate: int, block_size: int) -> None:
        super().__init__(sample_rate, block_size)
        self._sos = signal.butter(
            2, [LOW_HZ, HIGH_HZ], btype="bandpass", fs=sample_rate, output="sos"
        )
        self._zi = np.zeros((self._sos.shape[0], 2), dtype=np.float64)
        self._rng = np.random.default_rng()
        self._noise = np.empty(block_size, dtype=np.float32)
        self._drive = SmoothedValue(1.0, self.ramp_samples, block_size)
        self._makeup = SmoothedValue(1.0, self.ramp_samples, block_size)
        self._noise_amp = SmoothedValue(0.0, self.ramp_samples, block_size)
        self._out = np.empty(block_size, dtype=np.float32)

    def reset(self) -> None:
        self._zi.fill(0.0)

    def _apply_params(self, params: Mapping[str, float]) -> None:
        gain = MAX_DRIVE_GAIN ** params["drive"]  # 1..20, even steps in dB
        self._drive.set_target(gain)
        self._makeup.set_target(1.0 / float(np.tanh(gain)))
        self._noise_amp.set_target(MAX_NOISE_AMPLITUDE * params["noise"])

    def _process(self, block: np.ndarray) -> np.ndarray:
        n = len(block)
        noise_amp = self._noise_amp.next_block(n)
        if not (isinstance(noise_amp, float) and noise_amp == 0.0):
            noise = self._noise[:n]
            self._rng.standard_normal(out=noise, dtype=np.float32)
            noise *= noise_amp
            block += noise

        filtered, self._zi = signal.sosfilt(self._sos, block, zi=self._zi)
        out = self._out[:n]
        # Soft clip, normalised so a full-scale input still peaks at ~1.
        np.multiply(filtered, self._drive.next_block(n), out=out, casting="same_kind")
        np.tanh(out, out=out)
        out *= self._makeup.next_block(n)
        return out
