"""Command-line entry point: ``python -m timbrel``."""

from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from collections.abc import Sequence

import sounddevice as sd

from timbrel import __version__
from timbrel.core.chain import EffectChain
from timbrel.core.effects import EFFECTS, Effect, NoiseGate
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
    parser.add_argument(
        "--effect",
        action="append",
        default=[],
        metavar="NAME[:PARAM=VALUE,...]",
        help="add an effect, e.g. --effect pitch:semitones=-5 (repeatable, applied in order)",
    )
    parser.add_argument(
        "--gate",
        metavar="PARAM=VALUE,...",
        help="noise gate settings, e.g. --gate threshold_db=-45,release_ms=200",
    )
    parser.add_argument(
        "--exclusive-mic",
        action="store_true",
        help="open the mic in exclusive mode: ~18 ms lower latency, "
        "but other apps can't use the mic while Timbrel runs",
    )
    parser.add_argument(
        "--list-effects", action="store_true", help="list effects and their parameters and exit"
    )
    return parser


def parse_params(text: str) -> dict[str, float]:
    """Parse "a=1,b=-2.5" into {"a": 1.0, "b": -2.5}."""
    params: dict[str, float] = {}
    for item in filter(None, (part.strip() for part in text.split(","))):
        name, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"expected PARAM=VALUE, got {item!r}")
        try:
            params[name.strip()] = float(value)
        except ValueError:
            raise ValueError(f"{name.strip()}: {value!r} is not a number") from None
    return params


def parse_effect(spec: str, sample_rate: int, block_size: int) -> Effect:
    name, _, param_text = spec.partition(":")
    cls = EFFECTS.get(name.strip().lower())
    if cls is None:
        raise ValueError(f"unknown effect {name!r}; choose from {', '.join(EFFECTS)}")
    effect = cls(sample_rate, block_size)
    effect.set_params(**parse_params(param_text))
    return effect


def build_chain(
    effect_specs: Sequence[str], gate_spec: str | None, sample_rate: int, block_size: int
) -> EffectChain:
    effects = [parse_effect(spec, sample_rate, block_size) for spec in effect_specs]
    chain = EffectChain(sample_rate, block_size, effects)
    if gate_spec:
        chain.gate.set_params(**parse_params(gate_spec))
    return chain


def format_effect_list() -> str:
    lines = ["Effects (chain order: gate -> your --effect list -> limiter):"]
    for cls in (NoiseGate, *EFFECTS.values()):
        label = f"{cls.name} (set with --gate)" if cls is NoiseGate else cls.name
        lines.append(f"  {label}")
        for pname, spec in cls.PARAMS.items():
            unit = f" {spec.unit}" if spec.unit else ""
            lines.append(
                f"      {pname:<14} {spec.min:g} to {spec.max:g}{unit} (default {spec.default:g})"
            )
    return "\n".join(lines)


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


def _is_exclusive(settings: sd.WasapiSettings | None) -> bool:
    return settings is not None and bool(settings._streaminfo.flags & sd._lib.paWinWasapiExclusive)


def start_engine(config: EngineConfig, chain: EffectChain) -> tuple[Engine, EngineConfig]:
    """Start the engine, falling back to shared mode on each side that refuses
    exclusive access (device busy or format unsupported)."""
    shared = windows.shared_settings()
    attempts = [
        config,
        dataclasses.replace(config, input_settings=shared),
        dataclasses.replace(config, input_settings=shared, output_settings=shared),
    ]
    for attempt in attempts[:-1]:
        engine = Engine(attempt, chain)
        try:
            engine.start()
            return engine, attempt
        except sd.PortAudioError:
            pass
    engine = Engine(attempts[-1], chain)
    engine.start()
    return engine, attempts[-1]


def run(args: argparse.Namespace) -> int:
    block_size = args.block_size
    chain = build_chain(args.effect, args.gate, DEFAULT_SAMPLE_RATE, block_size)
    wasapi = windows.query_wasapi_devices()
    mic = windows.resolve_device(args.input, "input", wasapi)
    out = windows.resolve_device(args.output, "output", wasapi)
    windows.check_route(mic, out)

    config = EngineConfig(
        input_device=mic.index,
        output_device=out.index,
        sample_rate=DEFAULT_SAMPLE_RATE,
        block_size=block_size,
        output_channels=min(2, out.max_output_channels),
        input_settings=windows.stream_settings(mic, exclusive=args.exclusive_mic),
        output_settings=windows.stream_settings(out),
    )
    engine, config = start_engine(config, chain)
    mic_mode = (
        "exclusive"
        if config.input_settings is not None and _is_exclusive(config.input_settings)
        else "shared"
    )
    out_mode = "exclusive" if _is_exclusive(config.output_settings) else "shared"
    if args.exclusive_mic and mic_mode == "shared":
        print("note: the mic refused exclusive mode (busy or format unsupported); using shared.")
    print(f"Input:  [{mic.index}] {mic.name} ({mic_mode})")
    print(f"Output: [{out.index}] {out.name} ({out_mode})")
    names = " -> ".join(["gate", *(fx.name for fx in chain.effects), "limiter"])
    print(f"Chain:  {names}")
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
        if args.list_effects:
            print(format_effect_list())
            return 0
        return run(args)
    except (windows.DeviceError, ValueError, sd.PortAudioError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
