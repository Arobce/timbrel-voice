import numpy as np
import pytest
import sounddevice as sd

from timbrel.core.engine import Engine, EngineConfig


def make_engine(block_size: int = 256, output_channels: int = 2) -> Engine:
    return Engine(
        EngineConfig(
            input_device=0,
            output_device=1,
            block_size=block_size,
            output_channels=output_channels,
        )
    )


def run_block(engine: Engine, indata: np.ndarray, channels: int, flags: int = 0) -> np.ndarray:
    outdata = np.full((len(indata), channels), np.nan, dtype=np.float32)
    engine._callback(indata, outdata, len(indata), None, sd.CallbackFlags(flags))
    return outdata


@pytest.mark.parametrize("channels", [1, 2])
def test_passthrough_is_identity(channels):
    engine = make_engine(output_channels=channels)
    rng = np.random.default_rng(0)
    indata = rng.uniform(-1, 1, size=(256, 1)).astype(np.float32)

    outdata = run_block(engine, indata, channels)

    for ch in range(channels):
        np.testing.assert_array_equal(outdata[:, ch], indata[:, 0])


def test_passthrough_over_many_blocks_matches_input():
    engine = make_engine()
    signal = np.sin(np.linspace(0, 200 * np.pi, 256 * 50)).astype(np.float32)

    out = [run_block(engine, block[:, None], 2)[:, 0] for block in signal.reshape(50, 256)]

    np.testing.assert_array_equal(np.concatenate(out), signal)
    assert engine.stats.blocks == 50


def test_short_block_is_handled():
    engine = make_engine()
    indata = np.ones((100, 1), dtype=np.float32)

    outdata = run_block(engine, indata, 2)

    assert np.all(outdata == 1.0)


def test_xruns_are_counted():
    engine = make_engine()
    silence = np.zeros((256, 1), dtype=np.float32)

    run_block(engine, silence, 2)
    run_block(engine, silence, 2, sd._lib.paOutputUnderflow)
    run_block(engine, silence, 2, sd._lib.paInputOverflow | sd._lib.paOutputUnderflow)

    stats = engine.stats
    assert stats.output_underflows == 2
    assert stats.input_overflows == 1
    assert stats.input_underflows == 0
    assert stats.output_overflows == 0
    assert stats.xruns == 3
    assert stats.blocks == 3


@pytest.mark.parametrize("block_size", [64, 127, 1025, 4096])
def test_block_size_out_of_range_is_rejected(block_size):
    with pytest.raises(ValueError, match="block size"):
        make_engine(block_size=block_size)


@pytest.mark.parametrize("block_size", [128, 256, 1024])
def test_block_size_in_range_is_accepted(block_size):
    make_engine(block_size=block_size)


def test_not_running_before_start():
    engine = make_engine()
    assert not engine.running
    assert engine.latency_ms is None


# --- monitor ring buffer and monitor-only mode --------------------------------

from timbrel.core.engine import MonitorRing  # noqa: E402


def test_ring_round_trip():
    ring = MonitorRing(size=1024, max_fill=512)
    ring.write(np.arange(100, dtype=np.float32))
    out = np.empty(100, np.float32)
    assert ring.read(out)
    np.testing.assert_array_equal(out, np.arange(100))


def test_ring_underflow_pads_with_silence():
    ring = MonitorRing(size=1024, max_fill=512)
    ring.write(np.ones(30, np.float32))
    out = np.full(50, 9.0, np.float32)
    assert not ring.read(out)
    assert np.all(out[:30] == 1.0) and np.all(out[30:] == 0.0)


def test_ring_wraps_around():
    ring = MonitorRing(size=64, max_fill=64)
    out = np.empty(40, np.float32)
    for k in range(10):
        block = np.full(40, k, np.float32)
        ring.write(block)
        assert ring.read(out)
        assert np.all(out == k)


def test_ring_skips_ahead_when_backed_up():
    ring = MonitorRing(size=4096, max_fill=256)
    for k in range(10):
        ring.write(np.full(100, k, np.float32))
    out = np.empty(100, np.float32)
    ring.read(out)
    # Old audio is dropped so monitor latency stays bounded.
    assert out[-1] == 9.0


def test_monitor_receives_processed_audio():
    engine = Engine(EngineConfig(input_device=0, output_device=1, monitor_device=2))
    engine._monitor = object()  # pretend the monitor stream is open
    engine.set_monitor(True)
    blocks = [np.full((256, 1), k / 10, np.float32) for k in range(6)]
    for block in blocks:  # enough to fill the ~20 ms cushion
        run_block(engine, block, 2)
    out = np.zeros((256, 2), np.float32)
    engine._monitor_callback(out, 256, None, sd.CallbackFlags())
    # Plays the oldest queued block first, on every channel.
    np.testing.assert_array_equal(out[:, 0], blocks[0][:, 0])
    np.testing.assert_array_equal(out[:, 1], blocks[0][:, 0])
    assert engine.stats.monitor_underflows == 0


def test_monitor_disabled_outputs_silence():
    engine = Engine(EngineConfig(input_device=0, output_device=1, monitor_device=2))
    engine._monitor = object()
    run_block(engine, np.ones((256, 1), np.float32), 2)
    out = np.full((256, 2), 7.0, np.float32)
    engine._monitor_callback(out, 256, None, sd.CallbackFlags())
    assert np.all(out == 0.0)


def test_input_only_callback_updates_meters():
    engine = Engine(EngineConfig(input_device=0, output_device=None))
    indata = np.zeros((256, 1), np.float32)
    indata[10] = -0.75
    engine._input_callback(indata, 256, None, sd.CallbackFlags())
    assert engine.stats.input_peak == pytest.approx(0.75)
    assert engine.stats.output_peak == pytest.approx(0.75)
    assert engine.stats.blocks == 1


def test_ring_prebuffers_before_playing():
    ring = MonitorRing(size=1024, max_fill=512, prebuffer=150)
    out = np.full(100, 9.0, np.float32)
    ring.write(np.ones(100, np.float32))
    assert ring.read(out)  # cushion not full yet: silence, not a dropout
    assert np.all(out == 0.0)
    ring.write(np.ones(100, np.float32))
    assert ring.read(out)
    assert np.all(out == 1.0)


# --- voice test capture and output mute -------------------------------------------


def test_capture_records_raw_mic_across_blocks():
    engine = make_engine()
    engine.start_capture(0.01)  # 480 samples
    assert engine.capture_progress == 0.0
    blocks = [np.full((256, 1), k + 1, np.float32) for k in range(3)]
    for block in blocks:
        run_block(engine, block, 2)
    assert engine.capture_progress == 1.0
    clip = engine.captured()
    assert len(clip) == 480
    assert np.all(clip[:256] == 1.0)
    assert np.all(clip[256:] == 2.0)


def test_no_capture_until_started():
    engine = make_engine()
    run_block(engine, np.ones((256, 1), np.float32), 2)
    assert engine.capture_progress is None
    assert len(engine.captured()) == 0


def test_muted_output_is_silent_but_meters_still_work():
    engine = make_engine()
    engine.output_muted = True
    out = run_block(engine, np.full((256, 1), 0.5, np.float32), 2)
    assert np.all(out == 0.0)
    assert engine.stats.input_peak == pytest.approx(0.5)
