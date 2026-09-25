# Contributing to Timbrel

Thanks for helping! This guide covers setting up, the rules the audio code has to follow, and how to add an effect.

## Setup

Windows 10/11 and Python 3.11+ are required.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[dev]"
.\.venv\Scripts\python -m timbrel
```

To test end to end, install [VB-Audio Virtual Cable](https://vb-audio.com/Cable/). Never commit or bundle it.

## Checks (must pass before a PR)

```powershell
.\.venv\Scripts\ruff check .
.\.venv\Scripts\ruff format --check .
.\.venv\Scripts\python -m pytest
```

The tests are offline: no audio devices are needed. Effects run on generated test signals and `tests/audio/voice.wav`, which is a synthetic clip made by `tests/audio/make_voice.py`, and the UI is tested offscreen with a fake engine. CI runs the same checks on `windows-latest`.

Useful scripts:

- `python scripts/latency_test.py --exclusive-capture` measures real latency through VB-Cable (loopback).
- `powershell scripts/build.ps1` builds `dist\Timbrel\` and the release zip (needs `pip install -e ".[build]"`).

## Code layout

```
src/timbrel/
  core/engine.py        audio streams, callbacks, meters, monitor ring buffer
  core/chain.py         gate -> effects -> limiter, crossfades, bypass
  core/params.py        thread-safe parameters and smoothing
  core/effects/         one file per effect
  presets/              preset model/library; builtin/*.json
  app.py                controller (no Qt): settings, presets, engine, auto-resume
  ui/                   PySide6 window, tray, hotkeys, dialogs
  platform/windows.py   everything Windows-specific (devices, hotkeys, autostart)
```

## Audio thread rules

The callbacks in `core/engine.py`, and everything they call (`EffectChain.process` and each effect's `process`), run on PortAudio's audio thread. In that code:

- No file I/O, logging, printing, network, or Qt calls.
- Preallocate buffers in `__init__`; avoid allocating per block (use `out=` arguments and slices of preallocated arrays).
- No locks. The UI publishes changes by swapping a reference: `Effect.set_params` stores a new immutable mapping, and the audio thread picks it up at the start of the next block.
- Smooth parameter changes (`SmoothedValue`, about 10 ms) to avoid zipper noise; effect swaps and bypass crossfade over 20 ms in the chain.
- Target: zero dropouts and under 70 ms of added latency, VB-Cable included.

## Adding an effect

1. Create `src/timbrel/core/effects/<name>.py` with a subclass of `Effect`:

   ```python
   class Tremolo(Effect):
       name = "tremolo"
       PARAMS = {
           "rate_hz": ParamSpec(default=5.0, min=0.5, max=20.0, unit="Hz"),
           "depth": ParamSpec(0.5, 0.0, 1.0),
       }

       def __init__(self, sample_rate: int, block_size: int) -> None:
           super().__init__(sample_rate, block_size)
           # preallocate buffers sized to block_size here
           self.reset()

       def _apply_params(self, params):  # new values arrived (audio thread)
           ...  # set SmoothedValue targets, recompute coefficients

       def _process(self, block):  # may modify block in place; return same length
           ...

       def reset(self):  # clear delay lines / filter state
           ...
   ```

2. Register it in `EFFECTS` in `core/effects/__init__.py`. It then appears in `--list-effects`, can be used in presets, and gets sliders in the UI automatically.
3. Add offline tests in `tests/test_effects.py`: the generic tests already run every registered effect over `voice.wav` (same length, finite values, peak within the limiter ceiling through the chain). Add behaviour tests for what the effect actually does.
4. If the effect adds delay, set `self.latency_samples` so the latency readout includes it.

## Presets

Built-in presets are JSON files in `src/timbrel/presets/builtin/`: a `gate` plus an ordered `effects` list. The limiter is always appended. Keep the Clean preset matching the meeting spec in `docs/PRD.md`.

## Ground rules

- Keep Windows-specific code in `platform/windows.py`.
- Type hints everywhere; format with ruff.
- Small, focused commits with clear messages.
- Don't add GPL dependencies (the app is MIT). LGPL libraries used as separate files, like Qt, are fine.
- Never commit VB-Cable, voice models of real people, or copyrighted audio.
