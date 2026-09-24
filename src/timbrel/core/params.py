"""Thread-safe effect parameters and per-sample smoothing.

The UI thread publishes a new immutable parameter mapping by swapping one
reference; the audio thread reads that reference at the start of each block.
The audio thread never takes a lock.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np


@dataclass(frozen=True)
class ParamSpec:
    default: float
    min: float
    max: float
    unit: str = ""

    def clamp(self, value: float) -> float:
        return float(min(self.max, max(self.min, value)))


class ParamStore:
    """Latest parameter values, published by atomic reference swap."""

    def __init__(self, specs: Mapping[str, ParamSpec]) -> None:
        self._specs = dict(specs)
        self._write_lock = threading.Lock()  # serialises writers only
        self._current: Mapping[str, float] = MappingProxyType(
            {name: spec.default for name, spec in specs.items()}
        )

    def update(self, **params: float) -> None:
        unknown = set(params) - set(self._specs)
        if unknown:
            raise ValueError(
                f"unknown parameter(s) {', '.join(sorted(unknown))}; "
                f"expected one of {', '.join(self._specs)}"
            )
        clamped = {name: self._specs[name].clamp(value) for name, value in params.items()}
        with self._write_lock:
            self._current = MappingProxyType({**self._current, **clamped})

    def snapshot(self) -> Mapping[str, float]:
        """The current mapping. Identity changes whenever any value changes."""
        return self._current


class SmoothedValue:
    """A scalar that ramps linearly to each new target instead of jumping.

    ``next_block`` returns a float while settled (cheap, broadcasts in numpy
    expressions) and a per-sample ramp array while moving.
    """

    def __init__(self, value: float, ramp_samples: int, max_block: int) -> None:
        self.value = float(value)
        self.target = float(value)
        self._ramp_samples = max(1, int(ramp_samples))
        self._step = 0.0
        self._remaining = 0
        self._steps = np.arange(1, max_block + 1, dtype=np.float32)
        self._buf = np.empty(max_block, dtype=np.float32)

    def set_target(self, target: float) -> None:
        target = float(target)
        if target == self.target:
            return
        self.target = target
        self._remaining = self._ramp_samples
        self._step = (target - self.value) / self._ramp_samples

    def jump(self, value: float) -> None:
        self.value = self.target = float(value)
        self._remaining = 0

    @property
    def settled(self) -> bool:
        return self._remaining == 0

    def next_block(self, n: int) -> float | np.ndarray:
        if self._remaining == 0:
            return self.value
        k = min(n, self._remaining)
        out = self._buf[:n]
        np.multiply(self._steps[:k], self._step, out=out[:k])
        out[:k] += self.value
        out[k:] = self.target
        self._remaining -= k
        self.value = self.target if self._remaining == 0 else self.value + self._step * k
        return out
