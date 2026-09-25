"""Presets: a noise-gate setting plus an ordered list of effects.

Built-in presets ship as JSON in ``presets/builtin``; user presets are JSON
files in the user's app-data folder. The limiter is always appended by the
chain, so presets don't list it.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from timbrel.core.effects import EFFECTS, Effect, NoiseGate

BUILTIN_DIR = Path(__file__).with_name("builtin")
BUILTIN_ORDER = ["Clean", "Deep", "Chipmunk", "Robot", "Radio", "Demon", "Cave"]
MAX_NAME_LENGTH = 40


class PresetError(Exception):
    """A preset is invalid, or a save/rename/delete isn't allowed."""


@dataclass(frozen=True)
class EffectSpec:
    type: str
    params: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Preset:
    name: str
    effects: tuple[EffectSpec, ...] = ()
    gate: Mapping[str, float] = field(default_factory=dict)
    description: str = ""
    builtin: bool = False

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], builtin: bool = False) -> Preset:
        if not isinstance(data, Mapping):
            raise PresetError("preset must be a JSON object")
        name = validate_name(data.get("name", ""))
        gate = _check_params(NoiseGate.PARAMS, data.get("gate", {}), "gate")
        effects = []
        for i, item in enumerate(data.get("effects", [])):
            if not isinstance(item, Mapping) or item.get("type") not in EFFECTS:
                raise PresetError(f"effect {i + 1}: type must be one of {', '.join(EFFECTS)}")
            cls_ = EFFECTS[item["type"]]
            params = _check_params(cls_.PARAMS, item.get("params", {}), item["type"])
            effects.append(EffectSpec(item["type"], params))
        return cls(
            name=name,
            effects=tuple(effects),
            gate=gate,
            description=str(data.get("description", "")),
            builtin=builtin,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "gate": dict(self.gate),
            "effects": [{"type": e.type, "params": dict(e.params)} for e in self.effects],
        }

    def build_effects(self, sample_rate: int, block_size: int) -> list[Effect]:
        """Fresh effect instances with this preset's parameters."""
        effects = []
        for spec in self.effects:
            effect = EFFECTS[spec.type](sample_rate, block_size)
            effect.set_params(**spec.params)
            effects.append(effect)
        return effects

    def gate_params(self) -> dict[str, float]:
        """Full gate settings (defaults filled in), for NoiseGate.set_params."""
        return {name: self.gate.get(name, s.default) for name, s in NoiseGate.PARAMS.items()}


def _check_params(specs: Mapping[str, Any], params: Any, where: str) -> dict[str, float]:
    if not isinstance(params, Mapping):
        raise PresetError(f"{where}: params must be an object")
    unknown = set(params) - set(specs)
    if unknown:
        raise PresetError(f"{where}: unknown parameter(s) {', '.join(sorted(unknown))}")
    out = {}
    for name, value in params.items():
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise PresetError(f"{where}.{name}: must be a number")
        out[name] = specs[name].clamp(float(value))
    return out


def validate_name(name: Any) -> str:
    if not isinstance(name, str) or not name.strip():
        raise PresetError("preset name can't be empty")
    name = " ".join(name.split())
    if len(name) > MAX_NAME_LENGTH:
        raise PresetError(f"preset name must be at most {MAX_NAME_LENGTH} characters")
    return name


def _slug(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug or "preset"


def load_builtin() -> list[Preset]:
    presets = {}
    for path in BUILTIN_DIR.glob("*.json"):
        preset = Preset.from_dict(json.loads(path.read_text(encoding="utf-8")), builtin=True)
        presets[preset.name] = preset
    order = {name: i for i, name in enumerate(BUILTIN_ORDER)}
    return sorted(presets.values(), key=lambda p: (order.get(p.name, len(order)), p.name))


def write_json_atomic(path: Path, data: Any) -> None:
    """Write JSON so a crash mid-write never leaves a truncated file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.write("\n")
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


class PresetLibrary:
    """Built-in presets plus the user's own, saved as JSON files."""

    def __init__(self, user_dir: Path) -> None:
        self.user_dir = user_dir
        self._builtin = load_builtin()
        self._user: dict[str, tuple[Preset, Path]] = {}
        self.skipped: list[tuple[Path, str]] = []  # unreadable user files
        self.reload()

    def reload(self) -> None:
        self._user.clear()
        self.skipped.clear()
        if not self.user_dir.is_dir():
            return
        for path in sorted(self.user_dir.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                preset = Preset.from_dict(data)
            except (OSError, ValueError, PresetError) as exc:
                self.skipped.append((path, str(exc)))
                continue
            if self._find(preset.name) is None:
                self._user[preset.name.lower()] = (preset, path)

    def all(self) -> list[Preset]:
        user = sorted((p for p, _ in self._user.values()), key=lambda p: p.name.lower())
        return [*self._builtin, *user]

    def names(self) -> list[str]:
        return [p.name for p in self.all()]

    def get(self, name: str) -> Preset | None:
        return self._find(name)

    def _find(self, name: str) -> Preset | None:
        key = name.lower()
        for preset in self._builtin:
            if preset.name.lower() == key:
                return preset
        entry = self._user.get(key)
        return entry[0] if entry else None

    def save(self, preset: Preset, overwrite: bool = False) -> Preset:
        """Save a user preset. Built-in names are reserved."""
        name = validate_name(preset.name)
        existing = self._find(name)
        if existing is not None and existing.builtin:
            raise PresetError(f"{name!r} is a built-in preset; choose another name")
        if existing is not None and not overwrite:
            raise PresetError(f"a preset called {name!r} already exists")
        saved = Preset(name, preset.effects, dict(preset.gate), preset.description)
        path = self._user[name.lower()][1] if existing else self._free_path(name)
        write_json_atomic(path, saved.to_dict())
        self._user[name.lower()] = (saved, path)
        return saved

    def rename(self, old: str, new: str) -> Preset:
        preset = self._user_preset(old)
        new = validate_name(new)
        if new.lower() != old.lower() and self._find(new) is not None:
            raise PresetError(f"a preset called {new!r} already exists")
        _, old_path = self._user.pop(old.lower())
        renamed = Preset(new, preset.effects, dict(preset.gate), preset.description)
        path = old_path if _slug(new) == _slug(old) else self._free_path(new)
        write_json_atomic(path, renamed.to_dict())
        if path != old_path:
            old_path.unlink(missing_ok=True)
        self._user[new.lower()] = (renamed, path)
        return renamed

    def delete(self, name: str) -> None:
        self._user_preset(name)
        _, path = self._user.pop(name.lower())
        path.unlink(missing_ok=True)

    def _user_preset(self, name: str) -> Preset:
        preset = self._find(name)
        if preset is None:
            raise PresetError(f"no preset called {name!r}")
        if preset.builtin:
            raise PresetError(f"{preset.name!r} is built in and can't be changed")
        return preset

    def _free_path(self, name: str) -> Path:
        base = _slug(name)
        taken = {path for _, path in self._user.values()}
        path = self.user_dir / f"{base}.json"
        i = 2
        while path in taken or path.exists():
            path = self.user_dir / f"{base}-{i}.json"
            i += 1
        return path


def preset_from_effects(
    name: str, gate: NoiseGate, effects: Sequence[Effect], description: str = ""
) -> Preset:
    """Snapshot live effects (e.g. after slider edits) as a preset."""
    specs = tuple(EffectSpec(fx.name, dict(fx.params)) for fx in effects)
    return Preset(validate_name(name), specs, dict(gate.params), description)
