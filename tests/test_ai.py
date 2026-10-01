"""AI Voice: DSP helpers and streaming, tested with a fake converter (no GPU,
no onnxruntime needed)."""

import time

import numpy as np
import pytest
from scipy.signal import resample_poly

from timbrel.ai import VoiceFile, list_voices
from timbrel.ai.runtime import (
    GEN_FRAMES,
    PITCH_TAIL,
    RvcConverter,
    coarse_pitch,
    decode_f0,
    log_mel,
    mel_filterbank,
    smooth_f0,
)
from timbrel.ai.streaming import SAMPLE_RATE, StreamProcessor
from timbrel.ai.voice_effect import AiVoice, SpscRing
from timbrel.core.chain import EffectChain

SR = SAMPLE_RATE
HOP = SR // 10


class IdentityConverter:
    """Returns the input 'in the same voice': 16 kHz -> 40 kHz resample.
    Streaming output should then be the input, delayed and gain-adjusted."""

    def __init__(self):
        self.index_rate = 0.5
        self.calls = 0

    def convert(self, audio16, semitones, advance16=None):
        self.calls += 1
        # Like the real pipeline: HuBERT's 20 ms frames (x2 -> 100 fps) cover
        # 2 s minus the last 20 ms.
        n = min((len(audio16) // 320 - 1) * 2, GEN_FRAMES)
        return resample_poly(audio16, 5, 2).astype(np.float32)[: n * 400]


def tone(freq=220.0, seconds=3.0, amp=0.1):
    t = np.arange(int(seconds * SR)) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def noise(seconds=3.0, rms=0.1, seed=1):
    """Band-limited noise: unlike a tone it doesn't repeat, so a delay can be
    measured unambiguously."""
    from scipy.signal import butter, sosfilt

    x = np.random.default_rng(seed).standard_normal(int(seconds * SR))
    x = sosfilt(butter(6, 6000, fs=SR, output="sos"), x)
    return (x * rms / np.sqrt(np.mean(x**2))).astype(np.float32)


# --- pitch front end ---------------------------------------------------------------


def test_mel_filterbank_shape_and_order():
    fb = mel_filterbank()
    assert fb.shape == (128, 513)
    assert np.all(fb >= 0)
    peaks = fb.argmax(axis=1)
    assert np.all(np.diff(peaks) >= 0)  # filters rise in frequency


def test_log_mel_frame_rate_and_peak():
    t = np.arange(16000) / 16000
    mel = log_mel(np.sin(2 * np.pi * 1000 * t).astype(np.float32))
    assert mel.shape == (128, 101)  # 100 frames per second (+1 for centring)
    fb = mel_filterbank()
    band_1k = np.argmax(fb[:, round(1000 / (8000 / 512))])
    assert abs(int(np.median(mel.argmax(axis=0))) - band_1k) <= 1


def test_decode_f0_reads_the_peak_bin():
    cents = 1200 * np.log2(200 / 10)
    bin_ = int(round((cents - 1997.3794084376191) / 20))
    salience = np.zeros((3, 360), np.float32)
    salience[0, bin_] = 0.9
    salience[1, bin_] = 0.02  # below the voicing threshold
    f0 = decode_f0(salience)
    assert f0[0] == pytest.approx(200, rel=0.01)
    assert f0[1] == 0.0
    assert f0[2] == 0.0


def test_coarse_pitch_range():
    bins = coarse_pitch(np.array([0.0, 50.0, 200.0, 1100.0, 5000.0], np.float32))
    assert bins[0] == 1  # unvoiced
    assert bins[-1] == 255
    assert np.all(np.diff(bins[1:]) >= 0)


def test_smooth_f0_fixes_flicker_and_octave_jumps():
    f0 = np.full(40, 150.0, np.float32)
    f0[10] = 0.0  # one dropped frame
    f0[20] = 300.0  # one octave jump
    out = smooth_f0(f0, 5)
    assert out[10] == pytest.approx(150, rel=0.01)
    assert out[20] == pytest.approx(150, rel=0.01)
    silent = smooth_f0(np.zeros(20, np.float32), 5)
    assert np.all(silent == 0)


class LocalPitchModel:
    """Stands in for RMVPE: each frame's pitch bin follows its own mel column,
    so incremental and from-scratch runs must agree exactly."""

    def __init__(self):
        self.calls = 0

    def run(self, feeds):
        self.calls += 1
        window = feeds["input"]
        assert window.shape == (1, 128, PITCH_TAIL)  # fixed shape (CUDA graph)
        bins = (np.abs(window[0].mean(axis=0)) * 1000).astype(int) % 300 + 30
        out = np.zeros((1, PITCH_TAIL, 360), np.float32)
        out[0, np.arange(PITCH_TAIL), bins] = 1.0
        return out


def test_incremental_pitch_matches_from_scratch():
    class Base:
        rmvpe = LocalPitchModel()

    audio = noise(seconds=4.0)[::3].copy()  # 16 kHz
    window, step = 32000, 1600
    streaming = RvcConverter(Base(), voice=None)
    fresh = RvcConverter(Base(), voice=None)
    for i, end in enumerate(range(window, len(audio), step)):
        w = audio[end - window : end]
        got = streaming.pitch(w, None if i == 0 else step)
        # The first frames differ by design: from scratch they see the STFT's
        # mirrored padding; streaming kept their values from real audio.
        edge = 512 // 160 + 1 + 2  # padding reach + median smoothing
        np.testing.assert_array_equal(got[edge:], fresh.pitch(w)[edge:])
        fresh._salience = None
    calls = Base.rmvpe.calls
    streaming.pitch(audio[-window - step : -step], None)
    streaming.pitch(audio[-window:], step)
    assert Base.rmvpe.calls - calls == 4 + 1  # from scratch: 4 runs; a step: 1


# --- streaming -----------------------------------------------------------------------


def run_stream(processor, signal):
    out = [processor.process(signal[i : i + HOP]) for i in range(0, len(signal) - HOP + 1, HOP)]
    return np.concatenate(out)


def best_lag(a, b, max_lag):
    scores = [np.dot(a[: len(a) - lag], b[lag : len(a)]) for lag in range(max_lag)]
    return int(np.argmax(scores))


def test_stream_output_is_the_delayed_input():
    p = StreamProcessor(IdentityConverter())
    x = noise(rms=0.1)  # -20 dB RMS: the input gain stays at 0 dB
    y = run_stream(p, x)
    assert len(y) == len(x) // HOP * HOP
    lag = best_lag(x[SR:], y[SR:], SR // 5)
    assert abs(lag - p.delay_samples) <= p.search + 2
    aligned_x, aligned_y = x[SR : len(y) - lag], y[SR + lag :]
    corr = np.corrcoef(aligned_x, aligned_y[: len(aligned_x)])[0, 1]
    assert corr > 0.98


def test_stream_seams_are_smooth():
    p = StreamProcessor(IdentityConverter())
    y = run_stream(p, tone(amp=0.1))[SR:]
    steps = np.abs(np.diff(y))
    seams = steps[HOP - 1 :: HOP]
    assert seams.max() <= steps.max() * 1.05


def test_input_gain_lifts_quiet_speech_and_leaves_silence():
    p = StreamProcessor(IdentityConverter())
    quiet = tone(amp=0.003)  # about -53 dB RMS
    y = run_stream(p, quiet)
    assert p.gain_db == pytest.approx(30.0, abs=0.1)  # capped at +30 dB
    assert np.abs(y[SR:]).max() > np.abs(quiet).max() * 10
    p.reset()
    silent = run_stream(p, np.zeros(SR, np.float32))
    assert p.gain_db == 0.0
    assert np.abs(silent).max() == 0.0


def test_semitones_reach_the_converter():
    seen = []

    class Spy(IdentityConverter):
        def convert(self, audio16, semitones, advance16=None):
            seen.append(semitones)
            return super().convert(audio16, semitones)

    p = StreamProcessor(Spy())
    p.semitones = 7.0
    p.process(np.zeros(HOP, np.float32))
    assert seen == [7.0]


def test_converter_is_told_how_far_the_window_moved():
    seen = []

    class Spy(IdentityConverter):
        def convert(self, audio16, semitones, advance16=None):
            seen.append(advance16)
            return super().convert(audio16, semitones)

    p = StreamProcessor(Spy())
    for _ in range(3):
        p.process(np.zeros(HOP, np.float32))
    p.reset()  # the window's history is gone: the converter must start over
    p.process(np.zeros(HOP, np.float32))
    assert seen == [None, HOP // 3, HOP // 3, None]


# --- ring buffer and the effect ---------------------------------------------------------


def test_ring_round_trip_and_wrap():
    ring = SpscRing(10)
    out = np.empty(6, np.float32)
    for k in range(5):
        ring.write(np.full(6, k, np.float32))
        assert ring.read(out) == 6
        assert np.all(out == k)
    assert ring.read(out) == 0


def feed(processor, signal, ai, block=256):
    """Push blocks through ``processor`` (the effect or a chain holding it),
    letting the AI worker finish each conversion, as real-time pacing would."""
    out = []
    for i in range(0, len(signal) - block + 1, block):
        out.append(processor.process(signal[i : i + block].copy()).copy())
        deadline = time.monotonic() + 2
        while (ai._in.available() >= ai.processor.hop or ai.busy) and time.monotonic() < deadline:
            time.sleep(0.001)
    return np.concatenate(out)


@pytest.fixture
def effect():
    fx = AiVoice(SR, 256, StreamProcessor(IdentityConverter()), "test")
    yield fx
    fx.close()


def test_effect_is_silent_until_the_cushion_fills_then_plays(effect):
    y = feed(effect, tone(seconds=2.0), effect)
    first_sound = int(np.flatnonzero(np.abs(y) > 1e-4)[0])
    assert first_sound >= effect.processor.hop  # nothing before the first hop is ready
    assert np.abs(y[SR:]).max() > 0.05
    assert effect.underruns == 0


def test_effect_latency_includes_hop_processing_and_cushion(effect):
    p = effect.processor
    assert effect.latency_samples == p.hop + p.delay_samples + p.hop * 3 // 2


def test_effect_params_reach_processor_and_converter(effect):
    effect.set_params(semitones=12, index_rate=0.25)
    effect.process(np.zeros(256, np.float32))
    assert effect.processor.semitones == 12
    assert effect.processor.converter.index_rate == 0.25


def test_effect_reset_drops_queued_audio(effect):
    feed(effect, tone(seconds=1.0), effect)
    effect.reset()
    out = effect.process(np.zeros(256, np.float32))
    assert np.all(out == 0.0)


def test_effect_close_stops_the_worker():
    fx = AiVoice(SR, 256, StreamProcessor(IdentityConverter()), "test")
    fx.close()
    assert not fx._worker.is_alive()


def test_effect_runs_inside_the_chain_under_the_limiter(effect):
    chain = EffectChain(SR, 256, [effect])
    y = feed(chain, tone(seconds=2.0, amp=0.9), effect)
    assert np.all(np.isfinite(y))
    assert np.abs(y).max() <= 10 ** (-1 / 20) + 1e-6


# --- voice files -----------------------------------------------------------------------------


def test_list_voices_pairs_onnx_with_index(tmp_path):
    (tmp_path / "Alice.onnx").write_bytes(b"x")
    (tmp_path / "Alice.index").write_bytes(b"x")
    (tmp_path / "bob.onnx").write_bytes(b"x")
    (tmp_path / "notes.txt").write_text("x")
    voices = list_voices(tmp_path)
    assert voices == [
        VoiceFile("Alice", tmp_path / "Alice.onnx", tmp_path / "Alice.index"),
        VoiceFile("bob", tmp_path / "bob.onnx", None),
    ]
    assert list_voices(tmp_path / "missing") == []
