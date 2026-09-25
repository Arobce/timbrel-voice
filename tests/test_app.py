import pytest
import sounddevice as sd

from conftest import DEVICES, HOSTAPIS
from timbrel.app import Controller
from timbrel.core.engine import StreamStats
from timbrel.platform import windows
from timbrel.presets import PresetLibrary
from timbrel.settings import Settings


class FakeEngine:
    """Stands in for the audio engine: records config, never opens devices."""

    fail_exclusive = False
    fail_always = False

    def __init__(self, config, chain):
        self.config = config
        self.chain = chain
        self.stats = StreamStats()
        self.monitor_enabled = False
        self._running = False

    def start(self):
        exclusive = self.config.input_settings is not None and bool(
            self.config.input_settings._streaminfo.flags & sd._lib.paWinWasapiExclusive
        )
        if FakeEngine.fail_always or (FakeEngine.fail_exclusive and exclusive):
            raise sd.PortAudioError("device unavailable")
        self._running = True

    def stop(self):
        self._running = False

    @property
    def running(self):
        return self._running

    @property
    def has_monitor(self):
        return self.config.monitor_device is not None

    @property
    def latency_ms(self):
        return 55.0

    def set_monitor(self, enabled):
        self.monitor_enabled = enabled


@pytest.fixture(autouse=True)
def reset_fake():
    FakeEngine.fail_exclusive = False
    FakeEngine.fail_always = False


def make_controller(tmp_path, devices=DEVICES, settings=None, autostart=None):
    wasapi = windows.wasapi_devices(devices, HOSTAPIS)
    return Controller(
        settings or Settings(),
        tmp_path / "settings.json",
        PresetLibrary(tmp_path / "presets"),
        query_devices=lambda: wasapi,
        engine_factory=FakeEngine,
        autostart=autostart or (lambda enabled: None),
    )


def test_starts_on_mic_and_cable_with_clean(tmp_path):
    c = make_controller(tmp_path)
    assert c.start()
    cfg = c.engine.config
    assert cfg.input_device == 3  # system default mic
    assert cfg.output_device == 5  # CABLE Input
    assert c.preset.name == "Clean"
    assert [fx.name for fx in c.effects] == ["eq", "compressor"]
    status = c.status()
    assert status.running
    assert status.cable_found
    assert status.mic_exclusive


def test_exclusive_mic_falls_back_to_shared(tmp_path):
    FakeEngine.fail_exclusive = True
    c = make_controller(tmp_path)
    assert c.start()
    assert not c.status().mic_exclusive


def test_start_failure_is_reported_not_raised(tmp_path):
    FakeEngine.fail_always = True
    c = make_controller(tmp_path)
    assert not c.start()
    status = c.status()
    assert not status.running
    assert "device unavailable" in status.error


def test_monitor_only_without_cable(tmp_path):
    devices = [d for d in DEVICES if "CABLE" not in d["name"]]
    c = make_controller(tmp_path, devices=devices)
    assert c.start()
    assert c.engine.config.output_device is None
    assert c.engine.config.monitor_device == 4  # default speakers
    assert c.engine.monitor_enabled
    assert not c.status().cable_found


def test_remembers_devices_by_name(tmp_path):
    c = make_controller(tmp_path)
    c.start()
    c.set_devices("Microphone (Webcam)", "CABLE Input (VB-Audio Virtual Cable)", None)
    assert c.engine.config.input_device == 2
    assert Settings.load(tmp_path / "settings.json").input_device == "Microphone (Webcam)"


def test_missing_saved_device_falls_back_to_default(tmp_path):
    c = make_controller(tmp_path, settings=Settings(input_device="Unplugged Mic"))
    c.start()
    assert c.engine.config.input_device == 3


def test_select_preset_swaps_effects_and_persists(tmp_path):
    c = make_controller(tmp_path)
    c.start()
    c.select_preset("Robot")
    assert [fx.name for fx in c.effects] == ["robot", "eq"]
    assert c.gate.params["threshold_db"] == -48
    assert Settings.load(tmp_path / "settings.json").preset == "Robot"


def test_step_preset_wraps(tmp_path):
    c = make_controller(tmp_path)
    c.step_preset(-1)
    assert c.preset.name == "Cave"
    c.step_preset(1)
    assert c.preset.name == "Clean"


def test_set_param_marks_modified_and_save_as(tmp_path):
    c = make_controller(tmp_path)
    c.select_preset("Deep")
    c.set_param(0, "semitones", -7)
    c.set_param(None, "threshold_db", -40)
    assert c.modified
    saved = c.save_preset_as("Deeper")
    assert not c.modified
    assert c.preset.name == "Deeper"
    assert not saved.builtin
    reloaded = PresetLibrary(tmp_path / "presets").get("Deeper")
    assert reloaded.effects[0].params["semitones"] == -7
    assert reloaded.gate["threshold_db"] == -40


def test_rename_and_delete_current_preset(tmp_path):
    c = make_controller(tmp_path)
    c.save_preset_as("Mine")
    c.rename_preset("Mine", "Ours")
    assert c.preset.name == "Ours"
    c.delete_preset("Ours")
    assert c.preset.name == "Clean"


def test_bypass_persists_and_notifies(tmp_path):
    c = make_controller(tmp_path)
    calls = []
    c.listeners.append(lambda: calls.append(c.bypassed))
    c.toggle_bypass()
    assert c.chain.bypassed
    assert calls == [True]
    assert Settings.load(tmp_path / "settings.json").bypass is True


def test_block_size_change_rebuilds_chain_keeping_edits(tmp_path):
    c = make_controller(tmp_path)
    c.start()
    c.select_preset("Chipmunk")
    c.set_param(0, "semitones", 5)
    old_chain = c.chain
    c.set_block_size(512)
    assert c.chain is not old_chain
    assert c.chain.block_size == 512
    assert c.engine.config.block_size == 512
    assert c.effects[0].params["semitones"] == 5


def test_monitor_toggle(tmp_path):
    c = make_controller(tmp_path, settings=Settings(monitor_device="Headphones (USB Mic)"))
    c.start()
    c.set_monitor_enabled(True)
    assert c.engine.monitor_enabled
    c.set_monitor_enabled(False)
    assert not c.engine.monitor_enabled


def test_start_with_windows_calls_autostart(tmp_path):
    calls = []
    c = make_controller(tmp_path, autostart=calls.append)
    c.set_start_with_windows(True)
    assert calls == [True]
    assert Settings.load(tmp_path / "settings.json").start_with_windows


def test_unknown_saved_preset_falls_back(tmp_path):
    c = make_controller(tmp_path, settings=Settings(preset="Deleted One"))
    assert c.preset.name == "Clean"
