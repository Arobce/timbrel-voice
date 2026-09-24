"""Command-line entry point: ``python -m timbrel``."""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence

import sounddevice as sd

from timbrel import __version__
from timbrel.core.engine import (
    DEFAULT_BLOCK_SIZE,
    DEFAULT_SAMPLE_RATE,
    MAX_BLOCK_SIZE,
    MIN_BLOCK_SIZE,
    Engine,
    EngineConfig,
)
from timbrel.platform import windows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="timbrel",
        description="Real-time voice changer: mic -> effects -> VB-Audio Virtual Cable.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "--list-devices", action="store_true", help="list WASAPI input/output devices and exit"
    )
    parser.add_argument(
        "--input", metavar="DEVICE", help="input device index or name (default: system mic)"
    )
    parser.add_argument(
        "--output",
        metavar="DEVICE",
        help="output device index or name (default: CABLE Input, VB-Audio Virtual Cable)",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=DEFAULT_BLOCK_SIZE,
        metavar="N",
        help=f"samples per block, {MIN_BLOCK_SIZE}-{MAX_BLOCK_SIZE} (default: %(default)s)",
    )
    return parser


def format_device_list(wasapi: windows.WasapiDevices) -> str:
    cable = windows.find_cable(wasapi)
    lines: list[str] = []
    for direction, title in (("input", "Input devices"), ("output", "Output devices")):
        lines.append(f"{title} (WASAPI):")
        default = wasapi.default(direction)
        for device in wasapi.of(direction):
            tags = []
            if device.index == default:
                tags.append("default")
            if cable is not None and device.index == cable.index:
                tags.append("VB-Cable")
            suffix = f"  ({', '.join(tags)})" if tags else ""
            lines.append(f"  [{device.index:>3}] {device.name}{suffix}")
        lines.append("")
    if cable is None:
        lines.append(f"VB-Audio Virtual Cable not found. Install it from {windows.VB_CABLE_URL}")
    return "\n".join(lines).rstrip()


def run(input_spec: str | None, output_spec: str | None, block_size: int) -> int:
    wasapi = windows.query_wasapi_devices()
    mic = windows.resolve_device(input_spec, "input", wasapi)
    out = windows.resolve_device(output_spec, "output", wasapi)

    config = EngineConfig(
        input_device=mic.index,
        output_device=out.index,
        sample_rate=DEFAULT_SAMPLE_RATE,
        block_size=block_size,
        output_channels=min(2, out.max_output_channels),
        extra_settings=windows.stream_settings(),
    )
    engine = Engine(config)
    engine.start()
    print(f"Input:  [{mic.index}] {mic.name}")
    print(f"Output: [{out.index}] {out.name}")
    print(
        f"{config.sample_rate} Hz, block {block_size} "
        f"({block_size / config.sample_rate * 1000:.1f} ms), "
        f"latency {engine.latency_ms:.1f} ms. Press Ctrl+C to stop."
    )
    started = time.monotonic()
    try:
        while engine.running:
            time.sleep(0.5)
            elapsed = int(time.monotonic() - started)
            print(
                f"\r  {elapsed // 60:02d}:{elapsed % 60:02d}  xruns: {engine.stats.xruns}",
                end="",
                flush=True,
            )
    except KeyboardInterrupt:
        pass
    finally:
        engine.stop()
    print()
    stats = engine.stats
    print(
        f"Stopped. Blocks: {stats.blocks}, input overflows: {stats.input_overflows}, "
        f"input underflows: {stats.input_underflows}, "
        f"output overflows: {stats.output_overflows}, "
        f"output underflows: {stats.output_underflows}"
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.list_devices:
            print(format_device_list(windows.query_wasapi_devices()))
            return 0
        return run(args.input, args.output, args.block_size)
    except (windows.DeviceError, ValueError, sd.PortAudioError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
