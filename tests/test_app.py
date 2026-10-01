import numpy as np
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
        self.output_muted = False
        self.monitor_error = None
        self._running = False
        self._capture = None
        self.capture_progress = None

    def start_capture(self, seconds):
        # A loud 200 Hz "voice" arrives instantly in tests.
        t = np.arange(int(seconds * 48000)) / 48000
        self._capture = (0.3 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
        self.capture_progress = 1.0

    def captured(self):
        return self._capture.copy()

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


class FakePlayer:
    def __init__(self):
        self.played = []
        self.playing = False

    def play(self, clip, sample_rate, device):
        self.played.append((clip, device))
        self.playing = True

    def stop(self):
        self.playing = False


def make_controller(tmp_path, devices=DEVICES, settings=None, autostart=None, player=None):
    wasapi = windows.wasapi_devices(devices, HOSTAPIS)
    return Controller(
        settings or Settings(),
        tmp_path / "settings.json",
        PresetLibrary(tmp_path / "presets"),
        query_devices=lambda: wasapi,
        engine_factory=FakeEngine,
        autostart=autostart or (lambda enabled: None),
        player=player or FakePlayer(),
        ai_voices=lambda: [],
        ai_check=lambda: "test GPU",
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


def test_missing_saved_mic_pauses_instead_of_switching(tmp_path):
    c = make_controller(tmp_path, settings=Settings(input_device="Unplugged Mic"))
    assert not c.start()
    assert c.engine is None
    assert "Unplugged Mic" in c.status().error


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


# --- unplug detection and auto-resume ------------------------------------------


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def make_pluggable(tmp_path, settings=None):
    """Controller whose rescans return whatever ``state['devices']`` holds."""
    state = {"devices": DEVICES}
    clock = Clock()
    c = Controller(
        settings or Settings(),
        tmp_path / "settings.json",
        PresetLibrary(tmp_path / "presets"),
        query_devices=lambda: windows.wasapi_devices(state["devices"], HOSTAPIS),
        rescan_devices=lambda: windows.wasapi_devices(state["devices"], HOSTAPIS),
        engine_factory=FakeEngine,
        autostart=lambda enabled: None,
        clock=clock,
    )
    return c, state, clock


def test_first_start_remembers_chosen_devices(tmp_path):
    c = make_controller(tmp_path)
    c.start()
    saved = Settings.load(tmp_path / "settings.json")
    assert saved.input_device == "Microphone (USB Mic)"
    assert saved.output_device == "CABLE Input (VB-Audio Virtual Cable)"


def test_stalled_stream_pauses_then_resumes_when_device_returns(tmp_path):
    c, state, clock = make_pluggable(tmp_path)
    assert c.start()
    # Callbacks stop arriving (mic unplugged): paused after a few polls.
    for _ in range(4):
        c.poll()
    assert c.engine is None
    assert "stopped responding" in c.status().error

    # Still gone: the retry keeps it paused and explains why.
    state["devices"] = [d for d in DEVICES if d["name"] != "Microphone (USB Mic)"]
    clock.now += 3
    c.poll()
    assert c.engine is None
    assert "unplugged" in c.status().error

    # Plugged back in: resumes on the next retry.
    state["devices"] = DEVICES
    clock.now += 1
    c.poll()
    assert c.engine is None  # waits for the retry interval
    clock.now += 2
    c.poll()
    assert c.engine is not None
    assert c.status().running
    assert c.status().error is None


def test_poll_keeps_a_healthy_engine_running(tmp_path):
    c, _, _ = make_pluggable(tmp_path)
    c.start()
    for _ in range(10):
        c.engine.stats.blocks += 90  # audio is flowing
        c.poll()
    assert c.engine is not None


def test_explicit_no_output_is_monitor_only(tmp_path):
    c = make_controller(tmp_path, settings=Settings(output_device=""))
    assert c.start()
    assert c.engine.config.output_device is None
    assert c.engine.monitor_enabled


def test_hotkeys_persist(tmp_path):
    c = make_controller(tmp_path)
    c.set_hotkeys({"bypass": "ctrl+f9", "prev_preset": "", "next_preset": "f11"})
    saved = Settings.load(tmp_path / "settings.json")
    assert saved.hotkeys == {"bypass": "ctrl+f9", "prev_preset": "", "next_preset": "f11"}


# --- voice test ------------------------------------------------------------------


def test_voice_test_needs_running_audio(tmp_path):
    c = make_controller(tmp_path)
    assert not c.start_voice_test()
    assert c.voice_test_state() == ("idle", 0.0)


def test_voice_test_records_then_plays_with_effects(tmp_path):
    player = FakePlayer()
    c = make_controller(tmp_path, player=player)
    c.start()
    c.select_preset("Chipmunk")
    assert c.start_voice_test(seconds=1.0)
    assert c.voice_test_state()[0] == "ready"

    c.play_voice_test(processed=True)
    clip, device = player.played[-1]
    assert c.voice_test_state()[0] == "playing"
    assert c.engine.output_muted  # apps don't hear the speakers through the mic
    assert device == 4  # default speakers, never the cable
    assert len(clip) == 48000
    assert np.all(np.isfinite(clip))

    player.playing = False  # playback finished
    c.poll()
    assert not c.engine.output_muted
    assert c.voice_test_state()[0] == "ready"


def test_voice_test_original_is_unprocessed(tmp_path):
    c = make_controller(tmp_path)
    c.start()
    c.start_voice_test(seconds=0.5)
    c.voice_test_state()
    raw = c.render_voice_test(processed=False)
    np.testing.assert_array_equal(raw, c.engine.captured())


def test_voice_test_uses_current_preset_and_edits(tmp_path):
    c = make_controller(tmp_path)
    c.start()
    c.start_voice_test(seconds=1.0)
    c.voice_test_state()
    c.select_preset("Deep")
    deep = c.render_voice_test()
    c.set_param(0, "semitones", 12)
    up = c.render_voice_test()
    spectrum = lambda x: np.argmax(np.abs(np.fft.rfft(x[24000:])))  # noqa: E731
    assert spectrum(up) / spectrum(deep) == pytest.approx(2 ** (16 / 12), rel=0.05)


def test_voice_test_playback_prefers_monitor_device(tmp_path):
    player = FakePlayer()
    c = make_controller(
        tmp_path, settings=Settings(monitor_device="Headphones (USB Mic)"), player=player
    )
    c.start()
    c.start_voice_test(seconds=0.2)
    c.voice_test_state()
    c.play_voice_test()
    assert player.played[-1][1] == 7


def test_stop_playback_unmutes(tmp_path):
    player = FakePlayer()
    c = make_controller(tmp_path, player=player)
    c.start()
    c.start_voice_test(seconds=0.2)
    c.voice_test_state()
    c.play_voice_test()
    c.stop_test_playback()
    assert not player.playing
    assert not c.engine.output_muted


# --- monitor problems -------------------------------------------------------------


def test_missing_monitor_device_is_reported_but_audio_runs(tmp_path):
    settings = Settings(monitor_enabled=True, monitor_device="Unplugged Headphones")
    c = make_controller(tmp_path, settings=settings)
    assert c.start()
    status = c.status()
    assert status.running
    assert "Unplugged Headphones" in status.monitor_note
    assert "retry" in status.monitor_note


def test_monitor_that_failed_to_open_is_reported(tmp_path, monkeypatch):
    settings = Settings(monitor_enabled=True, monitor_device="Headphones (USB Mic)")
    c = make_controller(tmp_path, settings=settings)
    original = FakeEngine.start

    def start_without_monitor(self):
        original(self)
        self.monitor_error = "Device unavailable"

    monkeypatch.setattr(FakeEngine, "start", start_without_monitor)
    c.start()
    assert "Device unavailable" in c.status().monitor_note


def test_no_monitor_note_when_monitor_is_off(tmp_path):
    c = make_controller(tmp_path, settings=Settings(monitor_device="Unplugged Headphones"))
    c.start()
    assert c.status().monitor_note is None


def test_silent_monitor_is_restarted(tmp_path):
    c, state, clock = make_pluggable(
        tmp_path, settings=Settings(monitor_enabled=True, monitor_device="Headphones (USB Mic)")
    )
    c.start()
    first = c.engine
    first.set_monitor(True)
    for _ in range(4):
        first.stats.blocks += 90  # main audio keeps flowing
        c.poll()  # monitor_blocks never moves: headphones gone
    assert c.engine is not first  # restarted
    assert c.engine.running
    assert c.error is None


# --- AI voice in the controller (fake effect; no GPU) ------------------------------

import time as _time  # noqa: E402

from timbrel.core.effects.base import Effect  # noqa: E402
from timbrel.core.params import ParamSpec  # noqa: E402


class FakeAi(Effect):
    name = "ai_voice"
    PARAMS = {"semitones": ParamSpec(0.0, -24.0, 24.0), "index_rate": ParamSpec(0.5, 0.0, 1.0)}

    def __init__(self, sample_rate, block_size, voice="v"):
        super().__init__(sample_rate, block_size)
        self.voice = voice
        self.closed = False
        self.compute_ms, self.input_gain_db, self.underruns = 40.0, 12.0, 0

    def _apply_params(self, params):
        self.applied = dict(params)

    def _process(self, block):
        return block

    def reset(self):
        pass

    def close(self):
        self.closed = True


def make_ai_controller(tmp_path, fail=None, settings=None):
    loaded = []

    def loader(name, block_size):
        if fail:
            raise RuntimeError(fail)
        fx = FakeAi(48000, block_size, name)
        loaded.append(fx)
        return fx

    wasapi = windows.wasapi_devices(DEVICES, HOSTAPIS)
    c = Controller(
        settings or Settings(),
        tmp_path / "settings.json",
        PresetLibrary(tmp_path / "presets"),
        query_devices=lambda: wasapi,
        engine_factory=FakeEngine,
        autostart=lambda enabled: None,
        player=FakePlayer(),
        ai_loader=loader,
        ai_voices=lambda: ["Fp231", "Mp311"],
        ai_check=lambda: "test GPU",
    )
    return c, loaded


def wait_for_ai(c):
    deadline = _time.monotonic() + 3
    while c.ai_loading and _time.monotonic() < deadline:
        c.poll_ai()
        _time.sleep(0.01)
    c.poll_ai()


def finish_fades(c):
    for _ in range(10):  # let the chain's crossfades complete
        c.chain.process(np.zeros(256, np.float32))


def test_ai_voice_loads_in_background_then_replaces_the_preset(tmp_path):
    c, loaded = make_ai_controller(tmp_path)
    c.start()
    c.set_ai_voice("Fp231")
    assert c.ai_loading == "Fp231"
    assert not c.ai_active  # the classic preset keeps playing meanwhile
    wait_for_ai(c)
    assert c.ai_active
    assert c.chain.effects == (loaded[0],)
    assert c.gate.params["threshold_db"] == -55  # the AI gate
    status = c.status()
    assert status.ai_voice == "Fp231"
    assert status.ai_compute_ms == 40.0
    assert Settings.load(tmp_path / "settings.json").ai_voice == "Fp231"


def test_ai_settings_are_applied_and_saved(tmp_path):
    c, loaded = make_ai_controller(tmp_path)
    c.set_ai_param("semitones", 12)
    c.set_ai_voice("Fp231")
    wait_for_ai(c)
    c.set_ai_param("index_rate", 0.3)
    c.chain.process(np.zeros(256, np.float32))
    assert loaded[0].params == {"semitones": 12.0, "index_rate": 0.3}
    saved = Settings.load(tmp_path / "settings.json")
    assert (saved.ai_semitones, saved.ai_index_rate) == (12.0, 0.3)


def test_failed_load_shows_error_and_stays_classic(tmp_path):
    c, _ = make_ai_controller(tmp_path, fail="AI voice needs an NVIDIA GPU")
    c.set_ai_voice("Fp231")
    wait_for_ai(c)
    assert not c.ai_active
    assert "NVIDIA" in c.status().ai_error
    assert [fx.name for fx in c.effects] == ["eq", "compressor"]  # still Clean
    assert Settings.load(tmp_path / "settings.json").ai_voice is None


def test_turning_ai_off_restores_the_preset_and_releases_the_voice(tmp_path):
    c, loaded = make_ai_controller(tmp_path)
    c.set_ai_voice("Fp231")
    wait_for_ai(c)
    finish_fades(c)  # the AI voice is now playing
    c.set_ai_voice(None)
    assert [fx.name for fx in c.chain.effects] == ["eq", "compressor"]
    assert c.gate.params["threshold_db"] == -50  # Clean's gate again
    c.poll_ai()
    assert not loaded[0].closed  # still fading out in the chain
    finish_fades(c)
    c.poll_ai()
    assert loaded[0].closed


def test_picking_a_preset_leaves_ai_mode(tmp_path):
    c, loaded = make_ai_controller(tmp_path)
    c.set_ai_voice("Fp231")
    wait_for_ai(c)
    finish_fades(c)
    c.select_preset("Robot")
    assert not c.ai_active
    assert [fx.name for fx in c.effects] == ["robot", "eq"]
    finish_fades(c)
    c.poll_ai()
    assert loaded[0].closed


def test_switching_voice_mid_load_keeps_only_the_latest(tmp_path):
    c, loaded = make_ai_controller(tmp_path)
    c.set_ai_voice("First")
    c.set_ai_voice("Second")
    wait_for_ai(c)
    _time.sleep(0.1)
    c.poll_ai()
    assert c.chain.effects[0].voice == "Second"
    first = next(fx for fx in loaded if fx.voice == "First")
    assert first.closed


def test_presets_never_save_the_ai_voice(tmp_path):
    c, _ = make_ai_controller(tmp_path)
    c.set_ai_voice("Fp231")
    wait_for_ai(c)
    assert all(e.type != "ai_voice" for e in c.snapshot("Mine").effects)


def test_ai_voice_restored_on_startup(tmp_path):
    c, loaded = make_ai_controller(tmp_path, settings=Settings(ai_voice="Fp231"))
    wait_for_ai(c)
    assert c.ai_active
    assert loaded[0].voice == "Fp231"


def test_block_size_change_reloads_the_ai_voice(tmp_path):
    c, loaded = make_ai_controller(tmp_path)
    c.start()
    c.set_ai_voice("Fp231")
    wait_for_ai(c)
    c.set_block_size(512)
    wait_for_ai(c)
    assert c.ai_active
    assert loaded[-1].block_size == 512
    assert c.chain.effects == (loaded[-1],)


def test_shutdown_closes_ai_workers(tmp_path):
    c, loaded = make_ai_controller(tmp_path)
    c.set_ai_voice("Fp231")
    wait_for_ai(c)
    c.shutdown_ai()
    assert loaded[0].closed


def test_ai_voice_that_never_played_is_released_at_once(tmp_path):
    c, loaded = make_ai_controller(tmp_path)
    c.set_ai_voice("Fp231")
    wait_for_ai(c)
    c.set_ai_voice(None)  # before the audio thread ever used it
    c.poll_ai()
    assert loaded[0].closed
