"""Effect chain: noise gate -> effects -> limiter, with click-free switching.

Changing the effect list or toggling bypass crossfades over ~20 ms. Effects
coming in are first run silently over the last ~85 ms of gated input, so delay
lines and filters are already full and the new sound fades in without an onset.
The gate keeps running while bypassed (it's cheap) so that history, and its own
state, are current when effects come back. The UI
thread publishes changes by swapping one reference; the audio thread picks them
up at the start of a block. A change that arrives mid-crossfade waits until the
current crossfade finishes, so every transition is a clean two-way fade.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from timbrel.core.effects import Effect, Limiter, NoiseGate
from timbrel.core.params import SmoothedValue

CROSSFADE_MS = 20.0
HISTORY_SAMPLES = 4096  # ~85 ms at 48 kHz; longer than any effect's delay line


class EffectChain:
    def __init__(
        self,
        sample_rate: int,
        block_size: int,
        effects: Sequence[Effect] = (),
        bypass: bool = False,
    ) -> None:
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.gate = NoiseGate(sample_rate, block_size)
        self.limiter = Limiter(sample_rate, block_size)
        fade = int(sample_rate * CROSSFADE_MS / 1000)

        self._target_effects: tuple[Effect, ...] = tuple(effects)
        self._effects = self._target_effects
        self._old_effects: tuple[Effect, ...] = ()
        self._fade_len = fade
        self._fade_pos = fade  # == fade_len: not fading
        self._fade_ramp = np.arange(1, block_size + 1, dtype=np.float32) / fade

        self._bypass = bypass
        self._wet = SmoothedValue(0.0 if bypass else 1.0, fade, block_size)
        self._wet_was_silent = bypass

        self._wet_buf = np.empty(block_size, dtype=np.float32)
        self._old_buf = np.empty(block_size, dtype=np.float32)
        self._mix_buf = np.empty(block_size, dtype=np.float32)
        self._history = np.zeros(HISTORY_SAMPLES, dtype=np.float32)
        self._history_pos = 0  # oldest sample / next write
        self._prime_buf = np.empty(block_size, dtype=np.float32)

    # --- UI thread ---------------------------------------------------------

    def set_effects(self, effects: Sequence[Effect]) -> None:
        """Replace the effects between gate and limiter (use fresh instances)."""
        self._target_effects = tuple(effects)

    def set_bypass(self, bypass: bool) -> None:
        self._bypass = bypass

    @property
    def bypassed(self) -> bool:
        return self._bypass

    @property
    def effects(self) -> tuple[Effect, ...]:
        return self._target_effects

    @property
    def latency_samples(self) -> int:
        return sum(fx.latency_samples for fx in self._target_effects)

    # --- audio thread ------------------------------------------------------

    def reset(self) -> None:
        self.gate.reset()
        self.limiter.reset()
        for fx in self._effects:
            fx.reset()

    def process(self, block: np.ndarray) -> np.ndarray:
        n = len(block)
        self._wet.set_target(0.0 if self._bypass else 1.0)
        wet_gain = self._wet.next_block(n)

        wet = self._wet_buf[:n]
        np.copyto(wet, block)
        wet = self.gate.process(wet)

        if isinstance(wet_gain, float) and wet_gain == 0.0:
            # Fully bypassed: output is the dry input, untouched, and the
            # effects don't run (keeps CPU near zero).
            self._wet_was_silent = True
            if self._fade_pos < self._fade_len:
                self._fade_pos = self._fade_len
                self._old_effects = ()
            self._effects = self._target_effects
            self._remember(wet)
            return block

        if self._wet_was_silent:
            # Coming back from bypass: the effects' state is stale.
            self._effects = self._target_effects
            self.limiter.reset()
            self._prime(self._effects)
            self._wet_was_silent = False
        elif self._fade_pos >= self._fade_len and self._target_effects is not self._effects:
            self._old_effects = self._effects
            self._effects = self._target_effects
            self._prime(self._effects)
            self._fade_pos = 0
        self._remember(wet)

        wet = self._run_effects(wet)
        wet = self.limiter.process(wet)

        if isinstance(wet_gain, float) and wet_gain == 1.0:
            return wet
        # Bypass crossfade: out = dry + (wet - dry) * wet_gain
        out = self._mix_buf[:n]
        np.subtract(wet, block, out=out)
        out *= wet_gain
        out += block
        return out

    def _run_effects(self, wet: np.ndarray) -> np.ndarray:
        n = len(wet)
        if self._fade_pos >= self._fade_len:
            for fx in self._effects:
                wet = fx.process(wet)
            return wet

        old = self._old_buf[:n]
        np.copyto(old, wet)
        for fx in self._old_effects:
            old = fx.process(old)
        for fx in self._effects:
            wet = fx.process(wet)

        # Linear crossfade from old to new: wet = old + (wet - old) * ramp
        ramp = self._mix_buf[:n]
        np.add(self._fade_ramp[:n], self._fade_pos / self._fade_len, out=ramp)
        np.minimum(ramp, 1.0, out=ramp)
        if not np.shares_memory(wet, self._wet_buf):
            np.copyto(self._wet_buf[:n], wet)
            wet = self._wet_buf[:n]
        wet -= old
        wet *= ramp
        wet += old
        self._fade_pos += n
        if self._fade_pos >= self._fade_len:
            self._old_effects = ()
        return wet

    def _remember(self, block: np.ndarray) -> None:
        """Keep the last HISTORY_SAMPLES of gated input (ring buffer)."""
        n = len(block)
        pos = self._history_pos
        first = min(n, HISTORY_SAMPLES - pos)
        self._history[pos : pos + first] = block[:first]
        if first < n:
            self._history[: n - first] = block[first:]
        self._history_pos = (pos + n) % HISTORY_SAMPLES

    def _prime(self, effects: Sequence[Effect]) -> None:
        """Reset ``effects`` and run them silently over the recent input."""
        for fx in effects:
            fx.reset()
        if not effects:
            return
        chunk_len = self.block_size
        for start in range(0, HISTORY_SAMPLES, chunk_len):
            m = min(chunk_len, HISTORY_SAMPLES - start)
            chunk = self._prime_buf[:m]
            src = (self._history_pos + start) % HISTORY_SAMPLES
            first = min(m, HISTORY_SAMPLES - src)
            chunk[:first] = self._history[src : src + first]
            if first < m:
                chunk[first:] = self._history[: m - first]
            for fx in effects:
                chunk = fx.process(chunk)
