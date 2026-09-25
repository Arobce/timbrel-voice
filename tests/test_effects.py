"""Offline effect tests: no audio devices, just arrays and tests/audio/*.wav."""

from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from timbrel.core.chain import EffectChain
from timbrel.core.effects import (
    EFFECTS,
    Compressor,
    Echo,
    Eq,
    Limiter,
    NoiseGate,
    PitchShift,
    Radio,
    Robot,
)
from timbrel.core.params import SmoothedValue

SR = 48_000
BLOCK = 256
AUDIO_DIR = Path(__file__).parent / "audio"
ALL_EFFECTS = [NoiseGate, Limiter, *EFFECTS.values()]
CEILING = 10 ** (-1 / 20)


def load_wav(name: str) -> np.ndarray:
    sr, data = wavfile.read(AUDIO_DIR / name)
    assert sr == SR
    return (data.astype(np.float32) / 32768.0).astype(np.float32)


def run(processor, signal: np.ndarray, block: int = BLOCK, on_block=None) -> np.ndarray:
    """Feed ``signal`` through ``processor.process`` block by block."""
    out = []
    for i, start in enumerate(range(0, len(signal), block)):
        if on_block is not None:
            on_block(i)
        chunk = signal[start : start + block].copy()
        out.append(np.array(processor.process(chunk), dtype=np.float32, copy=True))
    return np.concatenate(out)


def sine(freq: float, seconds: float = 1.0, amp: float = 0.5) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def dominant_freq(signal: np.ndarray) -> float:
    windowed = signal * np.hanning(len(signal))
    spectrum = np.abs(np.fft.rfft(windowed, n=len(signal) * 4))
    return float(np.argmax(spectrum) * SR / (len(signal) * 4))


def band_rms(signal: np.ndarray) -> float:
    return float(np.sqrt(np.mean(signal**2)))


@pytest.fixture(scope="module")
def voice() -> np.ndarray:
    return load_wav("voice.wav")


# --- every effect on the sample clip ----------------------------------------


@pytest.mark.parametrize("cls", ALL_EFFECTS, ids=lambda c: c.name)
def test_effect_on_voice_keeps_length_and_is_finite(cls, voice):
    out = run(cls(SR, BLOCK), voice)
    assert out.shape == voice.shape
    assert np.all(np.isfinite(out))


@pytest.mark.parametrize("cls", list(EFFECTS.values()), ids=lambda c: c.name)
@pytest.mark.parametrize("gain", [1.0, 4.0])
def test_chain_output_never_exceeds_ceiling(cls, gain, voice):
    effect = cls(SR, BLOCK)
    # Push every parameter to its maximum for the worst case.
    effect.set_params(**{name: spec.max for name, spec in cls.PARAMS.items()})
    out = run(EffectChain(SR, BLOCK, [effect]), voice * gain)
    assert np.all(np.isfinite(out))
    assert np.abs(out).max() <= CEILING + 1e-6


@pytest.mark.parametrize("block", [128, 256, 1024])
def test_chain_handles_block_sizes(block, voice):
    chain = EffectChain(SR, block, [PitchShift(SR, block), Robot(SR, block), Radio(SR, block)])
    out = run(chain, voice, block=block)
    assert out.shape == voice.shape
    assert np.all(np.isfinite(out))


def test_bypass_is_identity(voice):
    chain = EffectChain(SR, BLOCK, [PitchShift(SR, BLOCK), Robot(SR, BLOCK)], bypass=True)
    np.testing.assert_array_equal(run(chain, voice), voice)


# --- noise gate --------------------------------------------------------------


def test_gate_silences_room_noise():
    noise = np.random.default_rng(0).normal(0, 10 ** (-65 / 20), SR).astype(np.float32)
    out = run(NoiseGate(SR, BLOCK), noise)
    assert np.abs(out).max() == 0.0


def test_gate_passes_speech_level_signal():
    tone = sine(300, amp=0.3)
    out = run(NoiseGate(SR, BLOCK), tone)
    np.testing.assert_allclose(out[SR // 10 :], tone[SR // 10 :], atol=1e-6)


def test_gate_fades_out_over_release_time():
    tone = np.concatenate([sine(300, 0.5, amp=0.3), np.zeros(SR // 2, np.float32)])
    tone[SR // 2 :] = sine(300, 0.5, amp=0.001)  # -60 dB tail, below the threshold
    gate = NoiseGate(SR, BLOCK)
    gate.set_params(release_ms=100)
    out = run(gate, tone)
    tail = out[SR // 2 :]
    # Still open during the hold time, then fades rather than cutting.
    assert np.abs(tail[: int(0.05 * SR)]).max() > 0
    assert np.abs(tail[int(0.45 * SR) :]).max() == 0.0


def test_gate_threshold_is_respected():
    tone = sine(300, amp=10 ** (-40 / 20) * np.sqrt(2))  # -40 dB RMS
    gate = NoiseGate(SR, BLOCK)
    gate.set_params(threshold_db=-30)
    assert np.abs(run(gate, tone)).max() == 0.0
    gate = NoiseGate(SR, BLOCK)
    gate.set_params(threshold_db=-50)
    assert np.abs(run(gate, tone)[SR // 10 :]).max() > 0.009


# --- limiter -----------------------------------------------------------------


def test_limiter_caps_loud_input():
    out = run(Limiter(SR, BLOCK), sine(440, amp=2.0))
    assert np.abs(out).max() <= CEILING + 1e-6


def test_limiter_leaves_quiet_input_alone():
    tone = sine(440, amp=0.5)
    np.testing.assert_array_equal(run(Limiter(SR, BLOCK), tone), tone)


def test_limiter_recovers_after_a_peak():
    signal = np.concatenate([sine(440, 0.2, amp=3.0), sine(440, 1.0, amp=0.5)])
    out = run(Limiter(SR, BLOCK), signal)
    assert np.abs(out[-SR // 4 :]).max() == pytest.approx(0.5, abs=0.01)


# --- robot -------------------------------------------------------------------


def test_robot_mix_zero_is_identity():
    robot = Robot(SR, BLOCK)
    robot.set_params(mix=0.0)
    robot._mix.jump(0.0)
    tone = sine(500)
    np.testing.assert_allclose(run(robot, tone), tone, atol=1e-6)


def test_robot_creates_sidebands():
    robot = Robot(SR, BLOCK)
    robot.set_params(freq_hz=100, mix=1.0)
    out = run(robot, sine(1000))[SR // 10 :]
    spectrum = np.abs(np.fft.rfft(out))
    freqs = np.fft.rfftfreq(len(out), 1 / SR)
    top_two = sorted(freqs[np.argsort(spectrum)[-2:]])
    assert top_two == pytest.approx([900, 1100], abs=3)


# --- radio -------------------------------------------------------------------


@pytest.mark.parametrize("freq", [80, 10_000])
def test_radio_band_limits(freq):
    def level(f: float) -> float:
        radio = Radio(SR, BLOCK)
        radio.set_params(drive=0.0, noise=0.0)
        return band_rms(run(radio, sine(f, amp=0.1))[SR // 10 :])

    assert level(freq) < level(1000) * 0.25


def test_radio_noise_adds_static_to_silence():
    radio = Radio(SR, BLOCK)
    radio.set_params(noise=1.0)
    assert band_rms(run(radio, np.zeros(SR, np.float32))) > 1e-3


def test_radio_output_bounded_for_full_scale_input():
    radio = Radio(SR, BLOCK)
    radio.set_params(drive=1.0, noise=0.0)
    assert np.abs(run(radio, sine(1000, amp=1.0))).max() <= 1.05


# --- pitch shift -------------------------------------------------------------


@pytest.mark.parametrize(
    ("semitones", "expected"),
    [
        (0, 220.0),
        (12, 440.0),
        (-12, 110.0),
        (7, 220.0 * 2 ** (7 / 12)),
        (-5, 220.0 * 2 ** (-5 / 12)),
    ],
)
def test_pitch_shift_moves_frequency(semitones, expected):
    shifter = PitchShift(SR, BLOCK)
    shifter.set_params(semitones=semitones)
    out = run(shifter, sine(220, seconds=2.0))[SR // 2 :]
    assert dominant_freq(out) == pytest.approx(expected, rel=0.02)


def test_pitch_shift_at_zero_is_a_pure_delay():
    shifter = PitchShift(SR, BLOCK)
    signal = np.random.default_rng(3).uniform(-0.5, 0.5, SR).astype(np.float32)
    out = run(shifter, signal)
    delay = shifter.latency_samples
    np.testing.assert_allclose(out[delay + BLOCK :], signal[BLOCK:-delay], atol=1e-5)


def test_pitch_shift_does_not_amplify(voice):
    shifter = PitchShift(SR, BLOCK)
    shifter.set_params(semitones=5)
    assert np.abs(run(shifter, voice)).max() <= np.abs(voice).max() + 1e-6


# --- click-free switching ----------------------------------------------------


def max_step(signal: np.ndarray) -> float:
    return float(np.abs(np.diff(signal)).max())


def test_switching_effects_crossfades():
    tone = sine(200, seconds=1.0, amp=0.5)
    natural = max_step(tone)
    chain = EffectChain(SR, BLOCK)

    def switch(i: int) -> None:
        if i == 50:
            chain.set_effects([PitchShift(SR, BLOCK)])  # hard switch = big jump

    out = run(chain, tone, on_block=switch)
    assert max_step(out) < natural * 3


def test_bypass_toggle_crossfades():
    tone = sine(200, seconds=1.0, amp=0.5)
    natural = max_step(tone)
    shifter = PitchShift(SR, BLOCK)
    shifter.set_params(semitones=-7)
    chain = EffectChain(SR, BLOCK, [shifter])

    def toggle(i: int) -> None:
        if i in (40, 80, 120):
            chain.set_bypass(not chain.bypassed)

    out = run(chain, tone, on_block=toggle)
    assert max_step(out) < natural * 3


def test_parameter_change_is_smoothed():
    tone = sine(200, seconds=1.0, amp=0.5)
    natural = max_step(tone)
    robot = Robot(SR, BLOCK)
    robot.set_params(mix=0.0)

    def change(i: int) -> None:
        if i == 50:
            robot.set_params(mix=1.0, freq_hz=150)

    out = run(robot, tone, on_block=change)
    assert max_step(out) < natural * 3


# --- params ------------------------------------------------------------------


def test_params_are_clamped():
    shifter = PitchShift(SR, BLOCK)
    shifter.set_params(semitones=40)
    assert shifter.params["semitones"] == 12.0


def test_unknown_param_raises():
    with pytest.raises(ValueError, match="unknown parameter"):
        Robot(SR, BLOCK).set_params(pitch=3)


def test_smoothed_value_ramps_then_settles():
    value = SmoothedValue(0.0, ramp_samples=100, max_block=64)
    value.set_target(1.0)
    first = value.next_block(64)
    assert isinstance(first, np.ndarray)
    assert first[0] == pytest.approx(0.01)
    assert first[-1] == pytest.approx(0.64)
    second = value.next_block(64)
    assert second[35] == pytest.approx(1.0)
    assert second[-1] == 1.0
    assert value.next_block(64) == 1.0


# --- eq ----------------------------------------------------------------------


def _eq(**params) -> Eq:
    eq = Eq(SR, BLOCK)
    eq.set_params(**params)
    return eq


def test_eq_low_cut_removes_rumble():
    rumble = band_rms(run(_eq(low_cut_hz=80), sine(30, amp=0.3))[SR // 5 :])
    voice_band = band_rms(run(_eq(low_cut_hz=80), sine(300, amp=0.3))[SR // 5 :])
    assert rumble < 0.3 / np.sqrt(2) * 0.2
    assert voice_band == pytest.approx(0.3 / np.sqrt(2), rel=0.05)


@pytest.mark.parametrize("gain_db", [-6.0, 2.0, 6.0])
def test_eq_presence_boost(gain_db):
    out = run(_eq(low_cut_hz=20, presence_db=gain_db), sine(3500, amp=0.1))[SR // 5 :]
    measured = 20 * np.log10(band_rms(out) / (0.1 / np.sqrt(2)))
    assert measured == pytest.approx(gain_db, abs=0.3)


# --- compressor --------------------------------------------------------------


def test_compressor_reduces_dynamic_range():
    comp = Compressor(SR, BLOCK)
    comp.set_params(threshold_db=-30, ratio=4)
    quiet = band_rms(run(comp, sine(300, amp=0.03))[SR // 2 :])
    comp.reset()
    loud = band_rms(run(comp, sine(300, amp=0.6))[SR // 2 :])
    # 26 dB input difference -> much less at the output.
    out_range = 20 * np.log10(loud / quiet)
    assert out_range < 15


def test_compressor_ratio_one_is_unity_after_makeup():
    comp = Compressor(SR, BLOCK)
    comp.set_params(ratio=1.0)
    tone = sine(300, amp=0.5)
    np.testing.assert_allclose(run(comp, tone), tone, atol=1e-6)


# --- echo --------------------------------------------------------------------


def test_echo_repeats_after_delay_time():
    echo = Echo(SR, BLOCK)
    echo.set_params(time_ms=100, feedback=0.0, mix=0.5)
    click = np.zeros(SR, np.float32)
    click[1000] = 1.0
    out = run(echo, click)
    assert out[1000] == pytest.approx(1.0)
    repeat = out[1000 + 4800 - 5 : 1000 + 4800 + 5]
    assert np.abs(repeat).max() > 0.3
    assert np.abs(out[1000 + 9600 - 50 :]).max() < 1e-6  # no feedback: single repeat


def test_echo_feedback_decays():
    echo = Echo(SR, BLOCK)
    echo.set_params(time_ms=100, feedback=0.5, mix=1.0)
    click = np.zeros(SR, np.float32)
    click[1000] = 1.0
    out = run(echo, click)
    peaks = [np.abs(out[1000 + k * 4800 - 20 : 1000 + k * 4800 + 20]).max() for k in (1, 2, 3)]
    assert peaks[0] > peaks[1] > peaks[2] > 0


def test_echo_mix_zero_is_dry():
    echo = Echo(SR, BLOCK)
    echo.set_params(mix=0.0)
    tone = sine(300)
    np.testing.assert_allclose(run(echo, tone), tone, atol=1e-6)
