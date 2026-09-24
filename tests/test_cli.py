import pytest

from timbrel.__main__ import (
    build_chain,
    build_parser,
    format_device_list,
    main,
    parse_effect,
    parse_params,
)
from timbrel.platform import windows


@pytest.fixture
def fake_devices(monkeypatch, wasapi):
    monkeypatch.setattr(windows, "query_wasapi_devices", lambda: wasapi)


def test_defaults():
    args = build_parser().parse_args([])
    assert args.input is None
    assert args.output is None
    assert args.block_size == 256
    assert not args.list_devices


def test_device_flags():
    args = build_parser().parse_args(["--input", "3", "--output", "cable", "--block-size", "512"])
    assert (args.input, args.output, args.block_size) == ("3", "cable", 512)


def test_device_list_marks_default_and_cable(wasapi):
    text = format_device_list(wasapi)
    assert "[  3] Microphone (USB Mic)  (default)" in text
    assert "[  5] CABLE Input (VB-Audio Virtual Cable)  (VB-Cable)" in text
    assert windows.VB_CABLE_URL not in text


def test_device_list_warns_when_cable_missing(wasapi_no_cable):
    assert windows.VB_CABLE_URL in format_device_list(wasapi_no_cable)


def test_list_devices_flag(fake_devices, capsys):
    assert main(["--list-devices"]) == 0
    assert "Input devices (WASAPI):" in capsys.readouterr().out


def test_bad_device_exits_with_error(fake_devices, capsys):
    assert main(["--input", "nonexistent"]) == 2
    assert "error:" in capsys.readouterr().err


def test_bad_block_size_exits_with_error(fake_devices, capsys):
    assert main(["--block-size", "64"]) == 2
    assert "block size" in capsys.readouterr().err


@pytest.mark.parametrize("flag", ["--help", "--version"])
def test_info_flags_exit_cleanly(flag):
    with pytest.raises(SystemExit) as exc:
        main([flag])
    assert exc.value.code == 0


def test_parse_params():
    assert parse_params("semitones=-5, mix=0.5") == {"semitones": -5.0, "mix": 0.5}
    assert parse_params("") == {}


@pytest.mark.parametrize("text", ["semitones", "semitones=low"])
def test_parse_params_rejects_bad_input(text):
    with pytest.raises(ValueError):
        parse_params(text)


def test_parse_effect_sets_params():
    effect = parse_effect("pitch:semitones=-5", 48000, 256)
    assert effect.name == "pitch"
    assert effect.params["semitones"] == -5.0


def test_parse_effect_unknown_name():
    with pytest.raises(ValueError, match="unknown effect"):
        parse_effect("vocoder", 48000, 256)


def test_build_chain_orders_effects_and_sets_gate():
    chain = build_chain(["robot", "pitch:semitones=3"], "threshold_db=-40", 48000, 256)
    assert [fx.name for fx in chain.effects] == ["robot", "pitch"]
    assert chain.gate.params["threshold_db"] == -40.0


def test_list_effects_flag(capsys):
    assert main(["--list-effects"]) == 0
    out = capsys.readouterr().out
    assert "semitones" in out
    assert "threshold_db" in out


def test_bad_effect_exits_with_error(fake_devices, capsys):
    assert main(["--effect", "pitch:semitones=abc"]) == 2
    assert "not a number" in capsys.readouterr().err
