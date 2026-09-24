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
