"""Full-duplex audio engine: mic in, processed mono out.

The callback runs on PortAudio's audio thread. It must not do file I/O, logging,
printing, network or Qt calls, and must not allocate per block.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import TracebackType
from typing import Any

import numpy as np
import sounddevice as sd

DEFAULT_SAMPLE_RATE = 48_000
DEFAULT_BLOCK_SIZE = 256
MIN_BLOCK_SIZE = 128
MAX_BLOCK_SIZE = 1024


@dataclass(frozen=True)
class EngineConfig:
    input_device: int
    output_device: int
    sample_rate: int = DEFAULT_SAMPLE_RATE
    block_size: int = DEFAULT_BLOCK_SIZE
    output_channels: int = 2
    extra_settings: Any = None

    def __post_init__(self) -> None:
        if not MIN_BLOCK_SIZE <= self.block_size <= MAX_BLOCK_SIZE:
            raise ValueError(
                f"block size must be between {MIN_BLOCK_SIZE} and {MAX_BLOCK_SIZE}, "
                f"got {self.block_size}"
            )
        if self.output_channels < 1:
            raise ValueError("output_channels must be at least 1")


class StreamStats:
    """Xrun counters written by the audio thread and read by other threads.

    Plain int increments are safe to read from another thread under the GIL;
    readers may see a value one block stale, which is fine for display.
    """

    def __init__(self) -> None:
        self.input_overflows = 0
        self.input_underflows = 0
        self.output_overflows = 0
        self.output_underflows = 0
        self.blocks = 0

    @property
    def xruns(self) -> int:
        return (
            self.input_overflows
            + self.input_underflows
            + self.output_overflows
            + self.output_underflows
        )


class Engine:
    """Mic -> (processing) -> output device, as one full-duplex stream."""

    def __init__(self, config: EngineConfig) -> None:
        self.config = config
        self.stats = StreamStats()
        self._mono = np.zeros(MAX_BLOCK_SIZE, dtype=np.float32)
        self._stream: sd.Stream | None = None

    def _callback(
        self,
        indata: np.ndarray,
        outdata: np.ndarray,
        frames: int,
        time: Any,
        status: sd.CallbackFlags,
    ) -> None:
        stats = self.stats
        if status:
            if status.input_overflow:
                stats.input_overflows += 1
            if status.input_underflow:
                stats.input_underflows += 1
            if status.output_overflow:
                stats.output_overflows += 1
            if status.output_underflow:
                stats.output_underflows += 1
        stats.blocks += 1

        mono = self._mono[:frames]
        np.copyto(mono, indata[:frames, 0])
        # Effects chain plugs in here (M1).
        outdata[:frames] = mono[:, np.newaxis]

    def start(self) -> None:
        if self._stream is not None:
            return
        cfg = self.config
        stream = sd.Stream(
            device=(cfg.input_device, cfg.output_device),
            samplerate=cfg.sample_rate,
            blocksize=cfg.block_size,
            dtype="float32",
            channels=(1, cfg.output_channels),
            latency="low",
            extra_settings=(cfg.extra_settings, cfg.extra_settings),
            callback=self._callback,
        )
        stream.start()
        self._stream = stream

    def stop(self) -> None:
        stream, self._stream = self._stream, None
        if stream is not None:
            stream.stop()
            stream.close()

    @property
    def running(self) -> bool:
        return self._stream is not None and self._stream.active

    @property
    def latency_ms(self) -> float | None:
        """Input + output latency reported by PortAudio, in milliseconds."""
        if self._stream is None:
            return None
        input_latency, output_latency = self._stream.latency
        return (input_latency + output_latency) * 1000.0

    def __enter__(self) -> Engine:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.stop()
