"""Pitch shift with two waveform-aligned delay-line taps (WSOLA-style).

Each tap reads the input through a variable delay that changes by
``1 - ratio`` samples per sample, so it plays back at ``ratio`` times real
speed: faster raises pitch, slower lowers it. A tap can only drift so far, so
when it reaches the edge of its grain it jumps by about one window and starts
a new grain. Two taps half a grain apart are crossfaded with sin^2 weights, so
each jump happens while that tap is silent.

A plain delay-line shifter jumps by a fixed amount, so the tap fading back in
is at an arbitrary point in the waveform relative to the tap it's mixed with:
the two partly cancel, heard as a phasey warble. Here the landing point is
chosen, near the ideal half-grain stagger, to line the waveform up with the
other tap (normalised cross-correlation), so the taps fade into each other in
phase.

Block-based with no FFT, it adds about half a window of delay. Formants move
with the pitch (the chipmunk / deep-voice sound); formant-preserving shift is
a separate effect.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

from timbrel.core.effects.base import Effect
from timbrel.core.params import ParamSpec, SmoothedValue

WINDOW_MS = 20.0
MATCH_MS = 8.0  # length of waveform compared when aligning a jump
LOWEST_PITCH_HZ = 70.0  # search covers +- half a period of this
MIN_DELAY = 2.0
BUFFER_SIZE = 8192  # power of two
EPSILON = 1e-9


class PitchShift(Effect):
    name = "pitch"
    PARAMS: Mapping[str, ParamSpec] = {
        "semitones": ParamSpec(0.0, -12.0, 12.0, "st"),
    }

    def __init__(self, sample_rate: int, block_size: int) -> None:
        super().__init__(sample_rate, block_size)
        self._window = sample_rate * WINDOW_MS / 1000
        self._match = int(sample_rate * MATCH_MS / 1000)
        self._search = int(sample_rate / LOWEST_PITCH_HZ / 2)
        self._max_delay = MIN_DELAY + 1.6 * self._window + self._search
        if self._max_delay + self._match + 2 * block_size >= BUFFER_SIZE:
            raise ValueError("block size too large for the pitch shifter's delay buffer")
        self.latency_samples = int(MIN_DELAY + self._window / 2)
        self._mask = BUFFER_SIZE - 1
        self._buf = np.zeros(BUFFER_SIZE, dtype=np.float64)
        self._ratio = SmoothedValue(1.0, self.ramp_samples, block_size)

        f64 = np.float64
        self._steps = np.arange(1, block_size + 1, dtype=f64)
        self._offsets = np.arange(block_size, dtype=f64)
        self._shift = np.empty(block_size, dtype=f64)  # cumulative delay change
        self._delays = [np.empty(block_size, dtype=f64) for _ in range(2)]
        self._weights = [np.empty(block_size, dtype=f64) for _ in range(2)]
        self._pos = np.empty(block_size, dtype=f64)
        self._frac = np.empty(block_size, dtype=f64)
        self._idx = np.empty(block_size, dtype=np.int64)
        self._idx1 = np.empty(block_size, dtype=np.int64)
        self._tap = np.empty(block_size, dtype=f64)
        self._tap0 = np.empty(block_size, dtype=f64)
        self._out = np.empty(block_size, dtype=f64)
        self._weight_sum = np.empty(block_size, dtype=f64)
        self.reset()

    def reset(self) -> None:
        self._buf.fill(0.0)
        self._write = 0
        w = self._window
        # Per tap: current delay and grain bounds [lo, hi]. Tap 1 starts half a
        # grain in, at full weight; tap 0 starts at a grain edge, silent.
        self._d = [MIN_DELAY, MIN_DELAY + w / 2]
        self._lo = [MIN_DELAY, MIN_DELAY]
        self._hi = [MIN_DELAY + w, MIN_DELAY + w]

    def _apply_params(self, params: Mapping[str, float]) -> None:
        self._ratio.set_target(2.0 ** (params["semitones"] / 12.0))

    def _process(self, block: np.ndarray) -> np.ndarray:
        n = len(block)
        head = self._write

        start = head & self._mask
        first = min(n, BUFFER_SIZE - start)
        self._buf[start : start + first] = block[:first]
        if first < n:
            self._buf[: n - first] = block[first:]

        shift = self._shift[:n]
        ratio = self._ratio.next_block(n)
        if isinstance(ratio, np.ndarray):
            np.subtract(1.0, ratio, out=shift)
            np.cumsum(shift, out=shift)
        else:
            np.multiply(self._steps[:n], 1.0 - ratio, out=shift)
        for tap in (0, 1):
            np.add(shift, self._d[tap], out=self._delays[tap][:n])

        self._handle_jumps(head, n)

        out, weight_sum = self._out[:n], self._weight_sum[:n]
        out.fill(0.0)
        weight_sum.fill(0.0)
        for tap in (0, 1):
            self._d[tap] = float(self._delays[tap][n - 1])
            self._read(tap, head, n, out, weight_sum)
        weight_sum += EPSILON
        out /= weight_sum

        self._write = head + n
        np.copyto(block, out, casting="same_kind")
        return block

    def _handle_jumps(self, head: int, n: int) -> None:
        """Apply both taps' jumps in time order, filling in grain weights."""
        seg_start = [0, 0]
        while True:
            next_jump = [self._next_crossing(tap, seg_start[tap], n) for tap in (0, 1)]
            tap = 0 if next_jump[0] <= next_jump[1] else 1
            k = next_jump[tap]
            if k >= n:
                break
            self._fill_weights(tap, seg_start[tap], k)
            self._jump(tap, head, k)
            seg_start[tap] = k
        for tap in (0, 1):
            self._fill_weights(tap, seg_start[tap], n)

    def _next_crossing(self, tap: int, start: int, n: int) -> int:
        seg = self._delays[tap][start:n]
        crossed = np.flatnonzero((seg < self._lo[tap]) | (seg > self._hi[tap]))
        return n if crossed.size == 0 else start + int(crossed[0])

    def _fill_weights(self, tap: int, a: int, b: int) -> None:
        """sin^2 window over the grain: silent at both edges, full in the middle."""
        if b <= a:
            return
        lo, hi = self._lo[tap], self._hi[tap]
        weight = self._weights[tap][a:b]
        np.subtract(self._delays[tap][a:b], lo, out=weight)
        weight *= math.pi / (hi - lo)
        np.clip(weight, 0.0, math.pi, out=weight)
        np.sin(weight, out=weight)
        np.square(weight, out=weight)

    def _jump(self, tap: int, head: int, k: int) -> None:
        """Start a new grain for ``tap`` at block sample ``k``, landing in phase
        with the other tap and about half a grain away from it."""
        delay = self._delays[tap]
        d = float(delay[k])
        d_other = float(self._delays[1 - tap][k])
        lo, hi, w = self._lo[tap], self._hi[tap], self._window
        going_up = d < lo  # delay shrinking (pitch up): jump to a longer delay

        # Ideal landing puts the other tap mid-grain: d_new = 2 * d_other - edge.
        if going_up:
            target = 2 * d_other - lo
            low, high = lo + 0.5 * w, lo + 1.6 * w
        else:
            target = 2 * d_other - hi
            low, high = max(MIN_DELAY, hi - 1.6 * w), hi - 0.5 * w
        high = min(high, self._max_delay)
        target = min(max(target, low), high)
        # Slide (don't clip) the search window into [low, high] so it always
        # spans a full period of the lowest pitch.
        centre = min(max(target, low + self._search), high - self._search)
        # Search offsets from the other tap (integer, so fractional phase is kept).
        dl_lo = math.ceil(max(centre - self._search, low) - d_other)
        dl_hi = math.floor(min(centre + self._search, high) - d_other)
        offset = self._best_offset(head + k - d_other, dl_lo, dl_hi, target - d_other)
        d_new = d_other + offset

        delay[k:] += d_new - d
        if going_up:
            self._hi[tap] = d_new
        else:
            self._lo[tap] = d_new

    def _best_offset(self, read_pos: float, dl_lo: int, dl_hi: int, fallback: float) -> float:
        """Delay offset in [dl_lo, dl_hi] whose waveform best matches the one at
        ``read_pos`` (normalised cross-correlation)."""
        if dl_hi <= dl_lo:
            return float(dl_lo) if dl_hi == dl_lo else fallback
        m = self._match
        p = math.floor(read_pos)
        ref = self._read_range(p - m + 1, p + 1)
        ref_energy = float(np.dot(ref, ref))
        if ref_energy < 1e-10:
            return fallback
        # Candidate windows end at p - offset, for offset = dl_hi .. dl_lo.
        region = self._read_range(p - dl_hi - m + 1, p - dl_lo + 1)
        corr = np.correlate(region, ref, mode="valid")
        energy = np.cumsum(np.square(region))
        energy = energy[m - 1 :] - np.concatenate(([0.0], energy[:-m]))
        score = corr / np.sqrt(np.maximum(energy, 1e-12) * ref_energy)
        return float(dl_hi - int(np.argmax(score)))

    def _read_range(self, start: int, stop: int) -> np.ndarray:
        return np.take(self._buf, np.arange(start, stop) & self._mask)

    def _read(self, tap: int, head: int, n: int, out: np.ndarray, weight_sum: np.ndarray) -> None:
        """Add this tap's weighted, linearly interpolated read to ``out``."""
        delay, weight = self._delays[tap][:n], self._weights[tap][:n]
        pos, frac = self._pos[:n], self._frac[:n]
        idx, idx1 = self._idx[:n], self._idx1[:n]
        value, value0 = self._tap[:n], self._tap0[:n]

        # Read position = the sample's own write index - delay.
        np.subtract(self._offsets[:n], delay, out=pos)
        pos += head + BUFFER_SIZE
        np.floor(pos, out=frac)
        np.copyto(idx, frac, casting="unsafe")
        np.subtract(pos, frac, out=frac)
        np.bitwise_and(idx, self._mask, out=idx)
        np.add(idx, 1, out=idx1)
        np.bitwise_and(idx1, self._mask, out=idx1)

        np.take(self._buf, idx, out=value0)
        np.take(self._buf, idx1, out=value)
        value -= value0
        value *= frac
        value += value0

        value *= weight
        out += value
        weight_sum += weight
