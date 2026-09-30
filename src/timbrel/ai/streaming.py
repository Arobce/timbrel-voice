"""Streaming voice conversion: 100 ms blocks in, 100 ms blocks out.

Each step converts the last 2 s of input (the models need context) and keeps
a 100 ms slice that ends ``lookahead`` before the newest audio (the model
does noticeably better with a little future context). Consecutive slices are
aligned by waveform matching (SOLA) and crossfaded, so the seams are smooth.
Runs on the AI worker thread, never the audio thread.
"""

from __future__ import annotations

import math
from typing import Protocol

import numpy as np
from scipy.signal import resample_poly

SAMPLE_RATE = 48000  # the engine's rate; resampling ratios below assume it
CONTEXT_SECONDS = 2.0
AGC_TARGET = 10 ** (-20 / 20)  # speech level fed to the models (RMS)
AGC_MAX = 10 ** (30 / 20)  # boost quiet mics by at most +30 dB
SPEECH_FLOOR = 10 ** (-70 / 20)  # below this a block is silence (the gate closed)


class Converter(Protocol):
    def convert(self, audio16: np.ndarray, semitones: float) -> np.ndarray:
        """2 s of 16 kHz audio -> the same span at 40 kHz in the target voice."""
        ...


class StreamProcessor:
    def __init__(
        self,
        converter: Converter,
        hop_ms: int = 100,
        crossfade_ms: int = 20,
        search_ms: int = 10,
        lookahead_ms: int = 100,
    ) -> None:
        sr = SAMPLE_RATE
        self.converter = converter
        self.semitones = 0.0
        self.hop = sr * hop_ms // 1000
        self.cf = sr * crossfade_ms // 1000
        self.search = sr * search_ms // 1000
        self.lookahead = sr * lookahead_ms // 1000
        self.context = np.zeros(int(CONTEXT_SECONDS * sr), np.float32)
        t = np.linspace(0, math.pi / 2, self.cf, dtype=np.float32)
        self.fade_in, self.fade_out = np.sin(t) ** 2, np.cos(t) ** 2
        self.reset()

    @property
    def delay_samples(self) -> int:
        """How far each output block trails its input block (processing only).
        HuBERT's 20 ms frames leave the newest ~20 ms out of the conversion."""
        return self.cf + self.search + self.lookahead + SAMPLE_RATE // 50

    def reset(self) -> None:
        self.context.fill(0.0)
        self.prev_tail = np.zeros(self.cf, np.float32)
        self.level: float | None = None  # tracked speech RMS for the input gain
        self.gain_db = 0.0

    def _input_gain(self, block: np.ndarray) -> float:
        """Bring speech to a steady level: quiet input loses voicing in the
        models. Silence (the chain's gate closed) doesn't move the estimate."""
        rms = float(np.sqrt(np.mean(block**2)))
        if rms > SPEECH_FLOOR:
            self.level = rms if self.level is None else 0.935 * self.level + 0.065 * rms
        gain = 1.0 if self.level is None else min(AGC_MAX, max(1.0, AGC_TARGET / self.level))
        self.gain_db = 20 * math.log10(gain)
        return gain

    def process(self, block: np.ndarray) -> np.ndarray:
        """One hop of 48 kHz input -> one hop of 48 kHz converted output."""
        n = len(block)
        gain = self._input_gain(block)
        self.context[:-n] = self.context[n:]
        np.clip(block * gain, -1.0, 1.0, out=self.context[-n:])

        audio16 = resample_poly(self.context, 1, 3).astype(np.float32)
        out40 = self.converter.convert(audio16, self.semitones)
        span = self.hop + self.cf + self.search + self.lookahead
        tail40 = out40[-(span * 5 // 6 + 64) :]
        out48 = resample_poly(tail40, 6, 5).astype(np.float32)
        chunk = out48[len(out48) - span : len(out48) - self.lookahead]

        cf = self.cf
        if np.any(self.prev_tail):
            # SOLA: start where the new chunk best continues the previous one.
            windows = np.lib.stride_tricks.sliding_window_view(chunk[: cf + self.search], cf)
            energy = np.sqrt(np.einsum("ij,ij->i", windows, windows)) + 1e-8
            offset = int(np.argmax((windows @ self.prev_tail) / energy))
        else:
            offset = 0
        out = chunk[offset : offset + self.hop].copy()
        out[:cf] = self.prev_tail * self.fade_out + out[:cf] * self.fade_in
        tail = chunk[offset + self.hop : offset + self.hop + cf]
        self.prev_tail = np.pad(tail, (0, cf - len(tail))).astype(np.float32)
        return np.clip(out, -1.0, 1.0)
