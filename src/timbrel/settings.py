"""User settings, persisted as JSON in the app-data folder."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from timbrel.core.engine import DEFAULT_BLOCK_SIZE, MAX_BLOCK_SIZE, MIN_BLOCK_SIZE
from timbrel.platform.windows import DEFAULT_HOTKEYS
from timbrel.presets import write_json_atomic


@dataclass
class Settings:
    # Devices are remembered by name: PortAudio indices change between boots.
    input_device: str | None = None
    output_device: str | None = None  # None = VB-Cable if present; "" = none
    monitor_device: str | None = None
    monitor_enabled: bool = False
    # Exclusive mic mode: ~18 ms lower latency; falls back to shared when busy.
    exclusive_mic: bool = True
    block_size: int = DEFAULT_BLOCK_SIZE
    preset: str = "Clean"
    bypass: bool = False
    start_with_windows: bool = False
    first_run_done: bool = False
    # AI Voice (experimental): the voice in use (None = off) and its settings.
    ai_voice: str | None = None
    ai_semitones: float = 0.0
    ai_index_rate: float = 0.5
    # action -> hotkey, e.g. {"bypass": "f9"}; empty string = unbound
    hotkeys: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_HOTKEYS))

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
        hotkeys = data.get("hotkeys")
        if isinstance(hotkeys, dict):
            for action in DEFAULT_HOTKEYS:
                if isinstance(hotkeys.get(action), str):
                    settings.hotkeys[action] = hotkeys[action]
        return settings

    def save(self, path: Path) -> None:
        write_json_atomic(path, asdict(self))


def _valid(name: str, value: Any) -> bool:
    if name in ("ai_semitones", "ai_index_rate"):
        return isinstance(value, int | float) and not isinstance(value, bool)
    if name in ("input_device", "output_device", "monitor_device", "ai_voice"):
        return value is None or isinstance(value, str)
    if name == "preset":
        return isinstance(value, str) and bool(value)
    if name == "block_size":
        return (
            isinstance(value, int)
            and not isinstance(value, bool)
            and MIN_BLOCK_SIZE <= value <= MAX_BLOCK_SIZE
        )
    if name == "hotkeys":
        return False  # merged separately, per action
    return isinstance(value, bool)
