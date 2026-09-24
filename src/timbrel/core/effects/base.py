"""Base class for all effects."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import ClassVar

import numpy as np

from timbrel.core.params import ParamSpec, ParamStore

# Ramp length for parameter changes, to avoid zipper noise.
PARAM_RAMP_MS = 10.0


class Effect(ABC):
    """A mono audio effect.

    ``set_params`` may be called from any thread. ``process`` and ``reset`` are
    called from the audio thread only: they must not block, do I/O or allocate
    large buffers. ``process`` may modify ``block`` in place and returns the
    processed block (same length, float32).
    """

    name: ClassVar[str]
    PARAMS: ClassVar[Mapping[str, ParamSpec]] = {}
    # Extra delay this effect adds, in samples (for the latency readout).
    latency_samples: int = 0

    def __init__(self, sample_rate: int, block_size: int) -> None:
        self.sample_rate = sample_rate
        self.block_size = block_size
        self._store = ParamStore(self.PARAMS)
        self._applied: Mapping[str, float] | None = None

    @property
    def ramp_samples(self) -> int:
        return int(self.sample_rate * PARAM_RAMP_MS / 1000)

    def set_params(self, **params: float) -> None:
        self._store.update(**params)

    @property
    def params(self) -> Mapping[str, float]:
        return self._store.snapshot()

    def process(self, block: np.ndarray) -> np.ndarray:
        params = self._store.snapshot()
        if params is not self._applied:
            self._applied = params
            self._apply_params(params)
        return self._process(block)

    @abstractmethod
    def _apply_params(self, params: Mapping[str, float]) -> None:
        """Take new parameter values (audio thread; set smoothing targets)."""

    @abstractmethod
    def _process(self, block: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def reset(self) -> None:
        """Clear internal state (delay lines, envelopes, filter memory)."""


def db_to_gain(db: float) -> float:
    return float(10.0 ** (db / 20.0))


def gain_to_db(gain: float) -> float:
    return float(20.0 * np.log10(max(gain, 1e-10)))
