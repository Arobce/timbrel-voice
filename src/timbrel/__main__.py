"""Command-line entry point: ``python -m timbrel``."""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence

import sounddevice as sd

from timbrel import __version__
from timbrel.app import _is_exclusive, start_engine
from timbrel.core.chain import EffectChain
from timbrel.core.effects import EFFECTS, Effect, NoiseGate
from timbrel.core.engine import (
    DEFAULT_BLOCK_SIZE,
    DEFAULT_SAMPLE_RATE,
    MAX_BLOCK_SIZE,
    MIN_BLOCK_SIZE,
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
    ai = parser.add_argument_group('AI voice (experimental; needs pip install -e ".[ai]")')
    ai.add_argument("--ai-voice", metavar="NAME", help="convert your voice with this AI voice")
    ai.add_argument(
        "--ai-shift", type=float, default=0.0, metavar="ST", help="AI voice pitch shift, semitones"
    )
    ai.add_argument(
        "--ai-index-rate",
        type=float,
        default=0.5,
        metavar="R",
        help="0-1: how strongly to sound like the voice (high values can garble words)",
    )
    ai.add_argument("--list-voices", action="store_true", help="list AI voices and exit")
    gui = parser.add_argument_group("window")
    gui.add_argument(
        "--no-gui", action="store_true", help="run in the terminal (implied by the audio options)"
    )
    gui.add_argument("--minimized", action="store_true", help="start minimized to the tray")
    gui.add_argument("--preset", metavar="NAME", help="preset to start with")
    return parser


def wants_cli(args: argparse.Namespace) -> bool:
    """Audio options (or --no-gui) run the terminal engine; otherwise the window."""
    return bool(
        args.no_gui
        or args.input
        or args.output
        or args.effect
        or args.gate
        or args.exclusive_mic
        or args.ai_voice
        or args.block_size != DEFAULT_BLOCK_SIZE
    )


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


def format_voice_list() -> str:
    from timbrel.ai import ai_dir, list_voices, voices_dir
    from timbrel.ai.runtime import HUBERT_FILE, RMVPE_FILE, AiUnavailable, check_runtime

    lines = [f"AI voices in {voices_dir()}:"]
    voices = list_voices()
    lines += [f"  {v.name}{'  (+ index)' if v.index else ''}" for v in voices] or ["  (none)"]
    missing = [f for f in (HUBERT_FILE, RMVPE_FILE) if not (ai_dir() / f).exists()]
    if missing:
        lines.append(f"Missing base models in {ai_dir()}: {', '.join(missing)}")
    try:
        lines.append(f"Runtime: {check_runtime()}")
    except AiUnavailable as exc:
        lines.append(f"Not available: {exc}")
    return "\n".join(lines)


def load_ai_voice(args: argparse.Namespace, block_size: int) -> Effect:
    from timbrel.ai import list_voices, load_voice

    voices = {v.name.lower(): v for v in list_voices()}
    voice = voices.get(args.ai_voice.lower())
    if voice is None:
        raise ValueError(f"no AI voice called {args.ai_voice!r}; see --list-voices")
    print(f"Loading AI voice {voice.name} on the GPU...")
    effect = load_voice(voice, DEFAULT_SAMPLE_RATE, block_size)
    effect.set_params(semitones=args.ai_shift, index_rate=args.ai_index_rate)
    return effect


def run(args: argparse.Namespace) -> int:
    block_size = args.block_size
    chain = build_chain(args.effect, args.gate, DEFAULT_SAMPLE_RATE, block_size)
    ai_effect = None
    if args.ai_voice:
        from timbrel.ai import AI_GATE

        ai_effect = load_ai_voice(args, block_size)
        chain.set_effects([*chain.effects, ai_effect])
        if not args.gate:
            chain.gate.set_params(**AI_GATE)
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
        if ai_effect is not None:
            ai_effect.close()
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
        if args.list_voices:
            print(format_voice_list())
            return 0
        if not wants_cli(args):
            from timbrel.ui import run_gui

            return run_gui(minimized=args.minimized, preset=args.preset)
        return run(args)
    except (windows.DeviceError, ValueError, sd.PortAudioError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
