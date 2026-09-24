"""Robot voice: ring modulation with a sine carrier."""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

from timbrel.core.effects.base import Effect
from timbrel.core.params import ParamSpec, SmoothedValue

TWO_PI = 2.0 * math.pi


class Robot(Effect):
    name = "robot"
    PARAMS: Mapping[str, ParamSpec] = {
        "freq_hz": ParamSpec(70.0, 30.0, 200.0, "Hz"),
        "mix": ParamSpec(1.0, 0.0, 1.0),
    }

    def __init__(self, sample_rate: int, block_size: int) -> None:
        super().__init__(sample_rate, block_size)
        defaults = {k: s.default for k, s in self.PARAMS.items()}
        self._freq = SmoothedValue(defaults["freq_hz"], self.ramp_samples, block_size)
        self._mix = SmoothedValue(defaults["mix"], self.ramp_samples, block_size)
        self._steps = np.arange(1, block_size + 1, dtype=np.float64)
        self._phase_buf = np.empty(block_size, dtype=np.float64)
        self._carrier = np.empty(block_size, dtype=np.float32)
        self.reset()

    def reset(self) -> None:
        self._phase = 0.0

    def _apply_params(self, params: Mapping[str, float]) -> None:
        self._freq.set_target(params["freq_hz"])
        self._mix.set_target(params["mix"])

    def _process(self, block: np.ndarray) -> np.ndarray:
        n = len(block)
        phase = self._phase_buf[:n]
        freq = self._freq.next_block(n)
        if isinstance(freq, np.ndarray):
            np.multiply(freq, TWO_PI / self.sample_rate, out=phase)
            np.cumsum(phase, out=phase)
        else:
            np.multiply(self._steps[:n], freq * TWO_PI / self.sample_rate, out=phase)
        phase += self._phase
        self._phase = float(phase[-1]) % TWO_PI

        carrier = self._carrier[:n]
        np.sin(phase, out=carrier, casting="same_kind")
        # y = x * ((1 - mix) + mix * carrier)
        mix = self._mix.next_block(n)
        carrier -= 1.0
        carrier *= mix
        carrier += 1.0
        block *= carrier
        return block
