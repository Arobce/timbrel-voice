"""Voice EQ: low cut (rumble) and presence boost (clarity)."""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
from scipy import signal

from timbrel.core.effects.base import Effect
from timbrel.core.params import ParamSpec

PRESENCE_HZ = 3500.0
PRESENCE_Q = 0.8


def peaking_sos(freq: float, gain_db: float, q: float, sample_rate: int) -> np.ndarray:
    """RBJ cookbook peaking EQ as one second-order section."""
    a = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * math.pi * freq / sample_rate
    alpha = math.sin(w0) / (2.0 * q)
    cos_w0 = math.cos(w0)
    b = [1 + alpha * a, -2 * cos_w0, 1 - alpha * a]
    den = [1 + alpha / a, -2 * cos_w0, 1 - alpha / a]
    return np.array([[*(x / den[0] for x in b), 1.0, den[1] / den[0], den[2] / den[0]]])


class Eq(Effect):
    name = "eq"
    PARAMS: Mapping[str, ParamSpec] = {
        "low_cut_hz": ParamSpec(80.0, 20.0, 300.0, "Hz"),
        "presence_db": ParamSpec(0.0, -6.0, 6.0, "dB"),
    }

    def __init__(self, sample_rate: int, block_size: int) -> None:
        super().__init__(sample_rate, block_size)
        self._sos = np.zeros((2, 6))
        self._zi = np.zeros((2, 2))

    def reset(self) -> None:
        self._zi.fill(0.0)

    def _apply_params(self, params: Mapping[str, float]) -> None:
        # Recomputed only when a parameter changes; filter state is kept, so a
        # slider drag doesn't click.
        self._sos[0] = signal.butter(
            2, params["low_cut_hz"], btype="highpass", fs=self.sample_rate, output="sos"
        )[0]
        self._sos[1] = peaking_sos(
            PRESENCE_HZ, params["presence_db"], PRESENCE_Q, self.sample_rate
        )[0]

    def _process(self, block: np.ndarray) -> np.ndarray:
        out, self._zi = signal.sosfilt(self._sos, block, zi=self._zi)
        np.copyto(block, out, casting="same_kind")
        return block
