import json

import pytest

from timbrel.platform import windows
from timbrel.settings import Settings


def test_defaults_when_missing(tmp_path):
    settings = Settings.load(tmp_path / "nope.json")
    assert settings == Settings()
    assert settings.preset == "Clean"
    assert settings.exclusive_mic is True


def test_round_trip(tmp_path):
    path = tmp_path / "Timbrel" / "settings.json"
    original = Settings(
        input_device="Microphone (fifine Microphone)",
        monitor_enabled=True,
        block_size=512,
        preset="Robot",
        first_run_done=True,
    )
    original.save(path)
    assert Settings.load(path) == original


def test_corrupt_file_gives_defaults(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text("{oops")
    assert Settings.load(path) == Settings()


def test_bad_values_are_ignored_individually(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(
        json.dumps(
            {"block_size": 4096, "bypass": "yes", "preset": "Deep", "unknown": 1, "input_device": 3}
        )
    )
    settings = Settings.load(path)
    assert settings.block_size == 256
    assert settings.bypass is False
    assert settings.input_device is None
    assert settings.preset == "Deep"


def test_find_by_name(wasapi):
    assert windows.find_by_name("Speakers (Realtek)", "output", wasapi).index == 4
    assert windows.find_by_name("Speakers (Realtek)", "input", wasapi) is None
    assert windows.find_by_name("Gone", "output", wasapi) is None
    assert windows.find_by_name(None, "output", wasapi) is None


def test_app_data_dir_uses_appdata(monkeypatch, tmp_path):
    monkeypatch.setenv("APPDATA", str(tmp_path))
    assert windows.app_data_dir() == tmp_path / "Timbrel"


class FakeWinreg:
    """Just enough of winreg for the autostart helpers, backed by a dict."""

    HKEY_CURRENT_USER = "HKCU"
    KEY_SET_VALUE = 2
    REG_SZ = 1

    def __init__(self):
        self.values = {}

    class _Key:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def OpenKey(self, root, path, reserved=0, access=0):  # noqa: N802
        assert path == windows.RUN_KEY
        return self._Key()

    def SetValueEx(self, key, name, reserved, kind, value):  # noqa: N802
        self.values[name] = value

    def DeleteValue(self, key, name):  # noqa: N802
        if name not in self.values:
            raise FileNotFoundError(name)
        del self.values[name]

    def QueryValueEx(self, key, name):  # noqa: N802
        if name not in self.values:
            raise FileNotFoundError(name)
        return self.values[name], self.REG_SZ


@pytest.fixture
def fake_registry(monkeypatch):
    fake = FakeWinreg()
    monkeypatch.setattr(windows, "winreg", fake)
    return fake


def test_autostart_toggle(fake_registry):
    assert not windows.autostart_enabled()
    windows.set_autostart(True, "timbrel.exe --minimized")
    assert windows.autostart_enabled()
    assert fake_registry.values["Timbrel"] == "timbrel.exe --minimized"
    windows.set_autostart(False)
    assert not windows.autostart_enabled()
    windows.set_autostart(False)  # removing twice is fine


def test_autostart_command_starts_minimized_with_clean():
    command = windows.autostart_command()
    assert "--minimized" in command
    assert '--preset "Clean"' in command
    assert "-m timbrel" in command
