"""Audio engine: mic in, processed mono out to the virtual cable and monitor.

The main stream is full duplex (mic -> cable). When monitoring is on, a second
output stream plays the same processed audio to headphones, fed through a
lock-free ring buffer. Without a cable the engine can run monitor-only.

Callbacks run on PortAudio's audio threads. They must not do file I/O,
logging, printing, network or Qt calls, and must not allocate per block.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import TracebackType
from typing import Any

import numpy as np
import sounddevice as sd

from timbrel.core.chain import EffectChain

DEFAULT_SAMPLE_RATE = 48_000
DEFAULT_BLOCK_SIZE = 256
MIN_BLOCK_SIZE = 128
MAX_BLOCK_SIZE = 1024


@dataclass(frozen=True)
class EngineConfig:
    input_device: int
    output_device: int | None
    sample_rate: int = DEFAULT_SAMPLE_RATE
    block_size: int = DEFAULT_BLOCK_SIZE
    output_channels: int = 2
    # Host-API specific settings (e.g. sd.WasapiSettings) for each device.
    input_settings: Any = None
    output_settings: Any = None
    monitor_device: int | None = None
    monitor_channels: int = 2
    monitor_settings: Any = None

    def __post_init__(self) -> None:
        if not MIN_BLOCK_SIZE <= self.block_size <= MAX_BLOCK_SIZE:
            raise ValueError(
                f"block size must be between {MIN_BLOCK_SIZE} and {MAX_BLOCK_SIZE}, "
                f"got {self.block_size}"
            )
        if self.output_channels < 1 or self.monitor_channels < 1:
            raise ValueError("channel counts must be at least 1")


class StreamStats:
    """Counters written by the audio threads and read by other threads.

    Plain int/float assignments are safe to read from another thread under the
    GIL; readers may see a value one block stale, which is fine for display.
    """

    def __init__(self) -> None:
        self.input_overflows = 0
        self.input_underflows = 0
        self.output_overflows = 0
        self.output_underflows = 0
        self.monitor_underflows = 0
        self.blocks = 0
        self.input_peak = 0.0
        self.output_peak = 0.0

    @property
    def xruns(self) -> int:
        """Dropouts on the main (mic -> cable) path."""
        return (
            self.input_overflows
            + self.input_underflows
            + self.output_overflows
            + self.output_underflows
        )


class MonitorRing:
    """Single-producer, single-consumer ring buffer for the monitor stream.

    The main callback writes, the monitor callback reads; each side only
    advances its own index, so no lock is needed. The two devices run on
    different clocks and wake at different times, so the reader keeps a small
    cushion: after starting (or running dry) it plays silence until
    ``prebuffer`` samples are queued. It skips ahead when too much builds up,
    keeping monitor latency bounded.
    """

    def __init__(self, size: int, max_fill: int, prebuffer: int = 0) -> None:
        self._buf = np.zeros(size, dtype=np.float32)
        self._size = size
        self._max_fill = max_fill
        self._prebuffer = prebuffer
        self._primed = prebuffer == 0
        self._write = 0  # total samples written
        self._read = 0  # total samples read

    def write(self, block: np.ndarray) -> None:
        n = len(block)
        start = self._write % self._size
        first = min(n, self._size - start)
        self._buf[start : start + first] = block[:first]
        if first < n:
            self._buf[: n - first] = block[first:]
        self._write += n

    def read(self, out: np.ndarray) -> bool:
        """Fill ``out``; returns False (and pads with silence) on underflow."""
        n = len(out)
        available = self._write - self._read
        if not self._primed:
            if available < self._prebuffer:
                out.fill(0.0)
                return True  # still filling the cushion; not a dropout
            self._primed = True
        if available > self._max_fill:
            self._read = self._write - max(self._prebuffer, n)
            available = self._write - self._read
        k = min(n, available)
        start = self._read % self._size
        first = min(k, self._size - start)
        out[:first] = self._buf[start : start + first]
        if first < k:
            out[first:k] = self._buf[: k - first]
        out[k:] = 0.0
        self._read += k
        if k < n:
            self._primed = self._prebuffer == 0
        return k == n

    def clear(self) -> None:
        self._read = self._write
        self._primed = self._prebuffer == 0


class Engine:
    """Mic -> effect chain -> cable (and optionally headphones)."""

    def __init__(self, config: EngineConfig, chain: EffectChain | None = None) -> None:
        self.config = config
        self.chain = chain
        self.stats = StreamStats()
        self.monitor_enabled = False
        self._mono = np.zeros(MAX_BLOCK_SIZE, dtype=np.float32)
        self._monitor_buf = np.zeros(MAX_BLOCK_SIZE * 4, dtype=np.float32)
        # WASAPI wakes each stream in bursts (e.g. two 256-sample callbacks
        # every ~10 ms), so the monitor keeps a ~20 ms cushion to ride them out.
        cushion = max(4 * config.block_size, config.sample_rate // 50)
        self._ring = MonitorRing(
            size=16 * MAX_BLOCK_SIZE,
            max_fill=cushion + 4 * config.block_size,
            prebuffer=cushion,
        )
        self._stream: sd.Stream | sd.InputStream | None = None
        self._monitor: sd.OutputStream | None = None

    # --- audio threads -----------------------------------------------------

    def _process(self, indata: np.ndarray, frames: int, status: sd.CallbackFlags) -> np.ndarray:
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
        stats.input_peak = float(max(mono.max(), -mono.min())) if frames else 0.0
        if self.chain is not None:
            mono = self.chain.process(mono)
        stats.output_peak = float(max(mono.max(), -mono.min())) if frames else 0.0
        if self.monitor_enabled and self._monitor is not None:
            self._ring.write(mono)
        return mono

    def _callback(
        self,
        indata: np.ndarray,
        outdata: np.ndarray,
        frames: int,
        time: Any,
        status: sd.CallbackFlags,
    ) -> None:
        mono = self._process(indata, frames, status)
        outdata[:frames] = mono[:, np.newaxis]

    def _input_callback(
        self, indata: np.ndarray, frames: int, time: Any, status: sd.CallbackFlags
    ) -> None:
        self._process(indata, frames, status)

    def _monitor_callback(
        self, outdata: np.ndarray, frames: int, time: Any, status: sd.CallbackFlags
    ) -> None:
        mono = self._monitor_buf[:frames]
        if not self.monitor_enabled:
            outdata.fill(0.0)
            return
        if not self._ring.read(mono):
            self.stats.monitor_underflows += 1
        outdata[:frames] = mono[:, np.newaxis]

    # --- control (UI / main thread) ----------------------------------------

    def start(self) -> None:
        if self._stream is not None:
            return
        cfg = self.config
        common = {
            "samplerate": cfg.sample_rate,
            "blocksize": cfg.block_size,
            "dtype": "float32",
            "latency": "low",
        }
        if cfg.output_device is not None:
            stream: sd.Stream | sd.InputStream = sd.Stream(
                device=(cfg.input_device, cfg.output_device),
                channels=(1, cfg.output_channels),
                extra_settings=(cfg.input_settings, cfg.output_settings),
                callback=self._callback,
                **common,
            )
        else:
            stream = sd.InputStream(
                device=cfg.input_device,
                channels=1,
                extra_settings=cfg.input_settings,
                callback=self._input_callback,
                **common,
            )
        monitor = None
        if cfg.monitor_device is not None:
            monitor = sd.OutputStream(
                device=cfg.monitor_device,
                channels=cfg.monitor_channels,
                extra_settings=cfg.monitor_settings,
                callback=self._monitor_callback,
                **common,
            )
        try:
            if monitor is not None:
                monitor.start()
            stream.start()
        except BaseException:
            for s in (stream, monitor):
                if s is not None:
                    s.close()
            raise
        self._stream, self._monitor = stream, monitor

    def stop(self) -> None:
        stream, self._stream = self._stream, None
        monitor, self._monitor = self._monitor, None
        for s in (stream, monitor):
            if s is not None:
                s.stop()
                s.close()
        self._ring.clear()

    def set_monitor(self, enabled: bool) -> None:
        if enabled and not self.monitor_enabled:
            self._ring.clear()
        self.monitor_enabled = enabled

    @property
    def running(self) -> bool:
        return self._stream is not None and self._stream.active

    @property
    def has_monitor(self) -> bool:
        return self._monitor is not None

    @property
    def latency_ms(self) -> float | None:
        """Stream latency reported by PortAudio plus effect delay, in ms.

        PortAudio's figure is an estimate; scripts/latency_test.py measures
        the real round trip.
        """
        if self._stream is None:
            return None
        latency = self._stream.latency
        total = sum(latency) if isinstance(latency, tuple) else latency
        effect_samples = self.chain.latency_samples if self.chain is not None else 0
        return (total + effect_samples / self.config.sample_rate) * 1000.0

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
