"""AI Voice as a chain effect: gate -> AI Voice -> limiter.

The audio thread only copies samples into one ring buffer and out of another.
A worker thread runs the model every hop (100 ms). Output starts after a
1.5-hop cushion: a new block only exists once the worker has converted it,
so the extra half hop covers the conversion time (~50 ms) and its jitter.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Mapping

import numpy as np

from timbrel.ai.streaming import StreamProcessor
from timbrel.core.effects.base import Effect
from timbrel.core.params import ParamSpec


class SpscRing:
    """Single-producer, single-consumer float ring. Each side advances only
    its own counter, so no lock is needed (int assignment is atomic)."""

    def __init__(self, size: int) -> None:
        self.buf = np.zeros(size, np.float32)
        self.size = size
        self.written = 0
        self.read_pos = 0

    def available(self) -> int:
        return self.written - self.read_pos

    def write(self, data: np.ndarray) -> None:
        n = len(data)
        start = self.written % self.size
        first = min(n, self.size - start)
        self.buf[start : start + first] = data[:first]
        if first < n:
            self.buf[: n - first] = data[first:]
        self.written += n

    def read(self, out: np.ndarray) -> int:
        """Fill ``out`` with up to len(out) samples; returns how many were read."""
        n = min(len(out), self.available())
        start = self.read_pos % self.size
        first = min(n, self.size - start)
        out[:first] = self.buf[start : start + first]
        if first < n:
            out[first:n] = self.buf[: n - first]
        self.read_pos += n
        return n

    def skip_to_end(self) -> None:
        """Reader only: drop everything queued."""
        self.read_pos = self.written


class AiVoice(Effect):
    name = "ai_voice"
    PARAMS: Mapping[str, ParamSpec] = {
        "semitones": ParamSpec(0.0, -24.0, 24.0, "st"),
        "index_rate": ParamSpec(0.5, 0.0, 1.0),
    }

    def __init__(
        self, sample_rate: int, block_size: int, processor: StreamProcessor, voice_name: str = ""
    ) -> None:
        super().__init__(sample_rate, block_size)
        self.processor = processor
        self.voice_name = voice_name
        hop = processor.hop
        self._in = SpscRing(hop * 8)
        self._out = SpscRing(hop * 8)
        self._cushion = hop + hop // 2
        self._primed = False
        self._block = np.zeros(block_size, np.float32)
        self.latency_samples = hop + processor.delay_samples + self._cushion
        self.underruns = 0  # output ran dry (worker too slow)
        self.compute_ms = 0.0  # last conversion time, for the UI
        self._reset_requested = False
        self.busy = False  # converting right now (lets tests wait for it)
        self._stop = threading.Event()
        self._worker = threading.Thread(target=self._run, name="timbrel-ai-voice", daemon=True)
        self._worker.start()

    # --- audio thread ------------------------------------------------------

    def _apply_params(self, params: Mapping[str, float]) -> None:
        # Plain attribute writes; the worker picks them up on its next hop.
        self.processor.semitones = params["semitones"]
        converter = self.processor.converter
        if hasattr(converter, "index_rate"):
            converter.index_rate = params["index_rate"]

    def _process(self, block: np.ndarray) -> np.ndarray:
        n = len(block)
        self._in.write(block)
        out = self._block[:n]
        if not self._primed:
            if self._out.available() < self._cushion:
                out.fill(0.0)
                return out
            self._primed = True
        got = self._out.read(out)
        if got < n:
            out[got:] = 0.0
            self.underruns += 1
            self._primed = False  # rebuild the cushion before playing again
        return out

    def reset(self) -> None:
        # Called on the audio thread (e.g. coming back from bypass): drop
        # queued output here, and ask the worker to drop its own state.
        self._out.skip_to_end()
        self._primed = False
        self._reset_requested = True

    # --- worker thread -----------------------------------------------------

    def _run(self) -> None:
        hop = self.processor.hop
        block = np.zeros(hop, np.float32)
        while not self._stop.is_set():
            if self._reset_requested:
                self._reset_requested = False
                self._in.skip_to_end()
                self.processor.reset()
            if self._in.available() < hop:
                time.sleep(0.002)
                continue
            self.busy = True
            self._in.read(block)
            start = time.perf_counter()
            converted = self.processor.process(block)
            self.compute_ms = (time.perf_counter() - start) * 1000
            self._out.write(converted)
            self.busy = False

    # --- control -----------------------------------------------------------

    def close(self) -> None:
        """Stop the worker (UI thread; the effect must be out of the chain)."""
        self._stop.set()
        self._worker.join(timeout=2)

    @property
    def input_gain_db(self) -> float:
        return self.processor.gain_db
