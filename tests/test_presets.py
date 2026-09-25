import json

import numpy as np
import pytest

from timbrel.core.chain import EffectChain
from timbrel.presets import (
    BUILTIN_ORDER,
    EffectSpec,
    Preset,
    PresetError,
    PresetLibrary,
    load_builtin,
    preset_from_effects,
)

SR, BLOCK = 48_000, 256


@pytest.fixture
def library(tmp_path):
    return PresetLibrary(tmp_path / "presets")


def test_all_builtins_load_in_order():
    assert [p.name for p in load_builtin()] == BUILTIN_ORDER
    assert all(p.builtin for p in load_builtin())


@pytest.mark.parametrize("preset", load_builtin(), ids=lambda p: p.name)
def test_builtin_runs_cleanly(preset):
    effects = preset.build_effects(SR, BLOCK)
    chain = EffectChain(SR, BLOCK, effects)
    chain.gate.set_params(**preset.gate_params())
    t = np.arange(SR) / SR
    voice = (0.4 * np.sin(2 * np.pi * 150 * t) * (1 + np.sin(2 * np.pi * 3 * t))).astype(np.float32)
    out = np.concatenate(
        [chain.process(voice[i : i + BLOCK].copy()).copy() for i in range(0, SR, BLOCK)]
    )
    assert np.all(np.isfinite(out))
    assert np.abs(out).max() <= 10 ** (-1 / 20) + 1e-6
    assert np.abs(out).max() > 0.05  # not silenced


def test_clean_matches_meeting_spec():
    clean = next(p for p in load_builtin() if p.name == "Clean")
    types = [e.type for e in clean.effects]
    assert "pitch" not in types
    params = {e.type: e.params for e in clean.effects}
    assert params["eq"]["low_cut_hz"] == pytest.approx(80)
    assert 1 <= params["eq"]["presence_db"] <= 3
    assert params["compressor"]["ratio"] == pytest.approx(3)
    assert clean.gate["threshold_db"] == pytest.approx(-50)


def test_round_trip_through_dict():
    preset = Preset("Mine", (EffectSpec("pitch", {"semitones": -3.0}),), {"threshold_db": -45.0})
    assert Preset.from_dict(preset.to_dict()) == preset


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ({"name": ""}, "empty"),
        ({"name": "x", "effects": [{"type": "vocoder"}]}, "type must be"),
        ({"name": "x", "effects": [{"type": "pitch", "params": {"cents": 1}}]}, "unknown"),
        ({"name": "x", "effects": [{"type": "pitch", "params": {"semitones": "a"}}]}, "number"),
        ({"name": "x", "gate": {"threshold_db": True}}, "number"),
    ],
)
def test_invalid_presets_are_rejected(data, message):
    with pytest.raises(PresetError, match=message):
        Preset.from_dict(data)


def test_out_of_range_params_are_clamped():
    preset = Preset.from_dict(
        {"name": "x", "effects": [{"type": "pitch", "params": {"semitones": 99}}]}
    )
    assert preset.effects[0].params["semitones"] == 12.0


def test_save_and_reload(library, tmp_path):
    library.save(Preset("My Voice", (EffectSpec("robot", {"mix": 0.5}),)))
    assert library.names()[-1] == "My Voice"
    reloaded = PresetLibrary(tmp_path / "presets")
    assert reloaded.get("my voice").effects[0].params == {"mix": 0.5}


def test_cannot_overwrite_builtin(library):
    with pytest.raises(PresetError, match="built-in"):
        library.save(Preset("clean"))


def test_duplicate_name_needs_overwrite(library):
    library.save(Preset("Mine"))
    with pytest.raises(PresetError, match="already exists"):
        library.save(Preset("mine"))
    library.save(Preset("Mine", (EffectSpec("robot"),)), overwrite=True)
    assert len(library.get("Mine").effects) == 1


def test_rename(library):
    library.save(Preset("Old Name"))
    library.rename("old name", "New Name")
    assert library.get("Old Name") is None
    assert library.get("new name") is not None
    files = list(library.user_dir.glob("*.json"))
    assert [f.name for f in files] == ["new-name.json"]


def test_rename_changing_only_case_keeps_file(library):
    library.save(Preset("mine"))
    library.rename("mine", "Mine")
    assert library.names()[-1] == "Mine"
    assert [f.name for f in library.user_dir.glob("*.json")] == ["mine.json"]


def test_rename_and_delete_refuse_builtins(library):
    with pytest.raises(PresetError, match="built in"):
        library.rename("Clean", "Other")
    with pytest.raises(PresetError, match="built in"):
        library.delete("Robot")


def test_delete(library):
    library.save(Preset("Temp"))
    library.delete("temp")
    assert library.get("Temp") is None
    assert not list(library.user_dir.glob("*.json"))


def test_names_with_same_slug_get_distinct_files(library):
    library.save(Preset("a b"))
    library.save(Preset("a-b"))
    assert sorted(f.name for f in library.user_dir.glob("*.json")) == ["a-b-2.json", "a-b.json"]


def test_corrupt_user_file_is_skipped(tmp_path):
    folder = tmp_path / "presets"
    folder.mkdir()
    (folder / "bad.json").write_text("{not json")
    (folder / "good.json").write_text(json.dumps({"name": "Good"}))
    library = PresetLibrary(folder)
    assert library.get("Good") is not None
    assert [p.name for p, _ in library.skipped] == ["bad.json"]


def test_preset_from_live_effects():
    preset = load_builtin()[1]  # Deep
    effects = preset.build_effects(SR, BLOCK)
    effects[0].set_params(semitones=-6)
    chain = EffectChain(SR, BLOCK, effects)
    snap = preset_from_effects("Deeper", chain.gate, effects)
    assert snap.effects[0] == EffectSpec("pitch", {"semitones": -6.0})
    assert set(snap.gate) == {"threshold_db", "release_ms"}
