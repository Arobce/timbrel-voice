"""User settings, persisted as JSON in the app-data folder."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from timbrel.core.engine import DEFAULT_BLOCK_SIZE, MAX_BLOCK_SIZE, MIN_BLOCK_SIZE
from timbrel.presets import write_json_atomic


@dataclass
class Settings:
    # Devices are remembered by name: PortAudio indices change between boots.
    input_device: str | None = None
    output_device: str | None = None
    monitor_device: str | None = None
    monitor_enabled: bool = False
    # Exclusive mic mode: ~18 ms lower latency; falls back to shared when busy.
    exclusive_mic: bool = True
    block_size: int = DEFAULT_BLOCK_SIZE
    preset: str = "Clean"
    bypass: bool = False
    start_with_windows: bool = False
    first_run_done: bool = False

    @classmethod
    def load(cls, path: Path) -> Settings:
        """Load settings; a missing or corrupt file gives defaults, and bad or
        unknown individual values are ignored."""
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return cls()
        if not isinstance(data, dict):
            return cls()
        settings = cls()
        for f in fields(cls):
            if f.name in data and _valid(f.name, data[f.name]):
                setattr(settings, f.name, data[f.name])
        return settings

    def save(self, path: Path) -> None:
        write_json_atomic(path, asdict(self))


def _valid(name: str, value: Any) -> bool:
    if name in ("input_device", "output_device", "monitor_device"):
        return value is None or isinstance(value, str)
    if name == "preset":
        return isinstance(value, str) and bool(value)
    if name == "block_size":
        return (
            isinstance(value, int)
            and not isinstance(value, bool)
            and MIN_BLOCK_SIZE <= value <= MAX_BLOCK_SIZE
        )
    return isinstance(value, bool)
