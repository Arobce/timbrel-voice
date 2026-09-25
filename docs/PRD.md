# Timbrel: Product Requirements

Full requirements for the project.
Work milestone by milestone (see "Milestones"). Do not start a later milestone until the current one meets its "Done when" criteria.

## Project summary

Timbrel is an open-source, real-time voice changer for Windows, for gaming and online meetings. It captures the mic, applies effects, and outputs to a virtual audio cable so Discord, Steam voice, in-game chat, Zoom, Teams and Google Meet hear the processed voice as a normal mic.

- **Platform:** Windows 10/11 (x64). macOS/Linux are future work; keep the core portable but do not implement them.
- **Language:** Python 3.11+
- **Distribution:** Open source on GitHub, downloadable as a zipped PyInstaller build from GitHub Releases.
- **Hardware:** Must run on CPU for classic effects. Optional AI voice mode (v2) uses an NVIDIA GPU via CUDA.

## Key decisions

| Decision | Choice | Notes |
| --- | --- | --- |
| Audio I/O | `sounddevice` (PortAudio, WASAPI host API) | Full-duplex stream, 48 kHz, float32, mono |
| DSP | `numpy`, `scipy.signal`, custom effects | Own pitch shifter, no `pedalboard` (see Decisions log) |
| UI | PySide6 (LGPL) | Single window + system tray |
| Hotkeys | `keyboard` or `pynput` | Global hotkeys that work while a game is focused |
| Config | JSON in `%APPDATA%\Timbrel\` | Settings + user presets |
| Packaging | PyInstaller, one-folder build, zipped | No installer in v1 |
| Virtual mic | VB-Audio Virtual Cable (user installs it) | Donationware; do NOT bundle or redistribute it, link to it |
| License | MIT | Decided in M1: own pitch shifter instead of GPL-3.0 `pedalboard` (see Decisions log) |

## Decisions log

### M1: Pitch shift is our own WSOLA-style delay line; license is MIT
- **Choice:** own implementation (`core/effects/pitch.py`), not `pedalboard`. `pedalboard` is GPL-3.0 and would make the whole app GPL; our own keeps the project MIT.
- **Algorithm:** two delay-line taps reading at `ratio` times real speed, crossfaded with sin² windows. When a tap reaches the end of its grain it jumps about one window, landing where normalised cross-correlation says its waveform lines up with the other tap, so the crossfade is in phase (WSOLA-style). A phase vocoder was rejected: it needs FFT frames of 20–40 ms plus overlap, blowing the latency budget.
- **Numbers:** 20 ms window, ~10 ms added delay, ~1% of one core. Harmonic-to-noise ratio of shifted harmonic tones (75–300 Hz, ±12 semitones) is 24–60 dB (mean ~50 dB); a fixed-jump delay line measured ~2–5 dB on the same test.
- **Trade-off:** formants move with pitch (chipmunk/deep sound). Formant correction is the separate P1 formant-shift effect.

### M1: Latency measurements
`scripts/latency_test.py` plays clicks into CABLE Input and records them from CABLE Output through a full-duplex stream like the engine's. This loop ≈ what Timbrel adds (our input buffering + our output buffering + VB-Cable's internal buffer), excluding effect delay.

| WASAPI mode (capture / playback) | Measured round trip |
| --- | --- |
| shared / shared | 93–117 ms |
| shared / exclusive (current: cable playback exclusive) | 67–78 ms |
| exclusive / exclusive | 51–55 ms |

PortAudio's reported latency is not reliable here (it reported 49 ms for the 93–117 ms case and 88 ms for the 51 ms case).

Further checks (VB-Cable internal rate 48 kHz):
- WASAPI exclusive on both sides stays at 52–55 ms whatever buffer size is requested (`latency="low"`, 10, 5, 3, 0 ms; block 128 or 256).
- Lowering VB-Cable's "Max Latency" from 7168 to 2048 samples (with a reboot) changed nothing: it is an upper bound, not the working buffer.
- Other host APIs on the same loop: WDM-KS 42–45 ms at 48 kHz (33–36 ms at 44.1 kHz); MME 170–190 ms; DirectSound 208–228 ms.

### M1: Latency target relaxed from < 40 ms to < 70 ms
With VB-Cable in the path, ~40 ms is the floor even on WDM-KS at 48 kHz with no effects, so < 40 ms is not reachable once pitch shift (~10 ms) is added. Decision: keep WASAPI (WDM-KS is fragile in PortAudio, locks devices, and saves only ~12 ms) and set the target to **< 70 ms added latency**. Measured: ~55 ms passthrough and ~65 ms with pitch shift using `--exclusive-mic`; ~70–80 ms passthrough in the default shared-mic mode. A WDM-KS backend (and 44.1 kHz) remains a possible future optimisation.

## Goals (v1)

- Real-time mic → effects → virtual cable with **< 70 ms added latency**, VB-Cable included (relaxed from < 40 ms in M1; see Decisions log)
- 6 built-in presets + adjustable sliders + saveable custom presets
- Global hotkeys: bypass toggle and preset cycling
- Monitor mode (hear yourself in headphones)
- A "Clean" preset good enough to leave on all day in meetings
- Zero crackles/underruns in a 60-minute session with a game running

## Non-goals (v1)

- AI voice conversion (that is v2, see below)
- Our own virtual audio driver
- Soundboard, recording, auto-update, code signing, installer
- macOS/Linux builds

## Functional requirements

### Audio I/O
- **FR1:** List input/output devices (WASAPI); remember the last choices.
- **FR2:** Auto-select "CABLE Input (VB-Audio Virtual Cable)" as output if present. If missing, show a banner with a link to https://vb-audio.com/Cable/ and still allow monitor-only use.
- **FR3:** Process mono at 48 kHz, float32. Default block size 256 samples (~5.3 ms), configurable 128–1024.
- **FR4:** Monitor toggle that also sends processed audio to a chosen headphone device (second output stream).

### Effects chain
Order: noise gate → selected effects → output limiter. Every effect implements:

```python
class Effect:
    def __init__(self, sample_rate: int, block_size: int): ...
    def set_params(self, **params) -> None: ...  # called from UI thread; must be thread-safe
    def process(
        self, block: np.ndarray
    ) -> np.ndarray: ...  # called from audio thread; no allocation where possible
    def reset(self) -> None: ...
```

| Effect | Params | Priority |
| --- | --- | --- |
| Noise gate | threshold dB (-70 to -20), release ms | P0 |
| Pitch shift | semitones -12 to +12 | P0 |
| Robot (ring modulator) | freq 30–200 Hz, mix 0–1 | P0 |
| Radio (band-pass 300–3400 Hz + soft clip + optional noise) | drive, noise level | P0 |
| Formant shift | -5 to +5 | P1 |
| Echo / reverb | time, feedback, mix | P1 |
| Output limiter | ceiling -1 dBFS | P0 (always on) |

### Presets
- **FR5:** Built-in: Clean, Deep, Chipmunk, Robot, Radio, Demon, Cave. Stored in `presets/builtin/*.json`. See "Meetings" for the Clean preset spec.
- **FR6:** Users can save, rename and delete custom presets (JSON in `%APPDATA%\Timbrel\presets\`).
- **FR7:** Switching presets crossfades over ~20 ms with no click or pop.

### Hotkeys
- **FR8:** Default F9 = bypass toggle, F10/F11 = previous/next preset. All rebindable in settings.
- **FR9:** Hotkeys must work while a full-screen game has focus.

### UI
- **FR10:** One window: input/output/monitor device pickers, preset list, effect sliders, big bypass button, input and output level meters, latency readout (ms), underrun counter.
- **FR11:** Close button minimizes to tray; tray menu has Bypass, Presets submenu, Quit.
- **FR12:** First-run dialog: explains VB-Cable install and how to pick "CABLE Output" as the mic in Discord, Zoom, Teams and Google Meet.
- **FR13:** Always-visible effect state: the window and the tray icon clearly show whether effects are ON or BYPASSED (different tray icon per state).

## Non-functional requirements

| Requirement | Target |
| --- | --- |
| Added latency | < 70 ms including VB-Cable, measured and shown in UI (see Decisions log) |
| Underruns | 0 in 60 minutes at default block size |
| CPU (classic effects) | < 10% of one core on a mid-range PC |
| CPU (bypassed or Clean preset) | < 3% of one core, so it can run all day alongside video calls |
| Memory | < 200 MB (classic mode) |
| Startup | Audio running within 3 s |

### Audio-thread rules (strict)
- No file I/O, logging, printing, network or Qt calls inside the audio callback.
- Pre-allocate buffers; avoid per-block allocation.
- UI → audio parameter changes go through a lock-free approach (atomic swap of a params object or a `queue.SimpleQueue` drained at the start of each callback).
- Smooth parameter changes over a few ms to avoid zipper noise.
- Count underruns/overflows from the callback `status` and expose the count to the UI.

## Architecture

```
mic ──► input stream ──► noise gate ──► effect chain ──► limiter ──► output stream ──► VB-Cable ──► Discord / game
                                                                          └──► monitor stream ──► headphones
UI + hotkeys ──(params only)──► effect chain
```

### Repo layout

```
timbrel/
├── README.md               # user-facing: download, setup VB-Cable, Discord setup, FAQ
├── LICENSE
├── CONTRIBUTING.md
├── pyproject.toml          # deps, ruff, pytest config
├── src/timbrel/
│   ├── __main__.py         # entry point
│   ├── core/
│   │   ├── engine.py       # streams, callback, chain, metrics
│   │   ├── chain.py        # effect chain + crossfade
│   │   ├── params.py       # thread-safe params
│   │   └── effects/        # one file per effect
│   ├── presets/            # load/save, builtin JSONs
│   ├── platform/windows.py # device detection, hotkeys, paths
│   ├── ai/                 # v2 only: AI voice conversion
│   └── ui/                 # PySide6 window, meters, tray, settings
├── tests/
│   ├── audio/              # sample WAVs (short, CC0)
│   └── test_effects.py
├── scripts/
│   ├── build.ps1           # PyInstaller build + zip
│   └── latency_test.py     # loopback latency measurement
└── .github/workflows/
    ├── ci.yml              # lint + tests on windows-latest
    └── release.yml         # build + attach zip on tag push
```

## Meetings

The app must work as an everyday mic processor for Zoom, Microsoft Teams, Google Meet, Slack huddles and Webex, not only for games.

### Clean preset
Goal: your real voice, just clearer. No pitch change.
- Noise gate (gentle, threshold around -50 dB, slow release so word endings aren't cut)
- High-pass filter at ~80 Hz to remove rumble
- Light presence boost (~+2 dB around 3–4 kHz)
- Gentle compressor (e.g. 3:1, auto makeup gain)
- Output limiter at -1 dBFS
Tune these by ear and by comparing against the raw mic on recorded test clips.

### Behaviour
- Bypass (F9) must switch instantly and silently, with no click, so users can drop back to their real voice mid-sentence.
- Option: "Start with Windows, minimized to tray, using Clean preset".
- Low CPU when bypassed or on Clean (see non-functional requirements).
- Latency matters less than in games; AI voice mode (v2) is acceptable in meetings.

### App setup (README section per app)
- **Zoom:** Settings → Audio → Microphone = "CABLE Output". Lower "Suppress background noise" or enable "Original sound for musicians" if effects sound choppy.
- **Microsoft Teams:** Settings → Devices → Microphone = "CABLE Output". Lower Noise suppression if effects are mangled.
- **Google Meet (browser):** Settings → Audio → Microphone = "CABLE Output"; allow mic access in the browser. Turn off Meet's noise cancellation if needed.
- **Slack huddles / Webex:** pick "CABLE Output" as the microphone in audio settings.
- Mention that meeting apps' own noise suppression can make robot/radio effects sound choppy.

### Responsible use
Add a short README note: people in work meetings usually expect to hear your real voice, and some workplaces restrict audio modification. Don't use voice effects to impersonate others or hide your identity where that's expected.

## Open source requirements

- **README.md:** what it is, screenshot/GIF, download link (Releases), VB-Cable setup, per-app setup for meetings (see "Meetings"), Discord setup (Input Device = "CABLE Output", turn off Krisp noise suppression if it muffles effects), hotkeys, troubleshooting, "check your game's rules on voice changers".
- **LICENSE:** GPL-3.0 or MIT per the decision above. Add a `THIRD_PARTY_NOTICES.md` listing dependency licenses.
- **CONTRIBUTING.md:** dev setup, how to add an effect, code style (ruff), tests must pass.
- **CI:** GitHub Actions on `windows-latest`: ruff + pytest (offline WAV tests only, no audio devices).
- **Releases:** pushing a tag `v*` builds with PyInstaller and attaches `Timbrel-vX.Y.Z-win64.zip` to the GitHub Release.
- **SmartScreen:** unsigned builds show a warning. Document "More info → Run anyway" in the README. Code signing is future work.
- Never commit or bundle VB-Cable, AI voice models of real people, or copyrighted audio.

## Testing

- **Offline (CI):** run every effect on sample WAVs; assert same length, no NaN/Inf, peak ≤ 1.0, bypass = identity, preset JSONs load.
- **Latency:** `scripts/latency_test.py` plays a click to the output and records it back through loopback; report round-trip ms.
- **Manual soak:** 60-minute Discord call with a game running; underrun counter stays at 0.
- **Compatibility checklist:** Discord, Steam voice, 2–3 games with in-game voice chat, Zoom, Microsoft Teams, Google Meet (Chrome and Edge).
- **Clean preset check:** record the same sentence raw and with Clean; Clean should sound clearer with no pumping, cut-off word endings or artifacts.

## Milestones

Commit at the end of each milestone. Run `pytest` after every change.

### M0: Passthrough
Build `core/engine.py` with a full-duplex `sounddevice` stream (WASAPI) from the chosen mic to VB-Cable, plus a CLI: `python -m timbrel --list-devices`, `--input`, `--output`.
**Done when:** voice reaches Discord through CABLE Output, unchanged, with no crackles for 10 minutes.

### M1: Classic effects
Effect interface, chain, noise gate, pitch shift, robot, radio, limiter. Offline tests. Choose the pitch-shift implementation (own phase vocoder/PSOLA vs `pedalboard`) based on latency and license, and record the decision in this file.
**Done when:** effects switchable via CLI flag, tests pass, latency < 70 ms (relaxed from < 40 ms; see Decisions log).

### M2: UI + presets
PySide6 window (FR10–FR13), presets (FR5–FR7) including the Clean preset, settings persistence, start-with-Windows option. Audio stays on its own thread.
**Done when:** everything controllable from the UI; presets save/load; no clicks on switch.

### M3: Hotkeys + polish
Global hotkeys (FR8–FR9), tray, latency readout, underrun counter, error states (missing cable, unplugged device → pause and auto-resume).
**Done when:** hotkeys work in a full-screen game; 60-minute soak passes in both a game and a Zoom or Teams call.

### M4: Open source release
README, LICENSE, CONTRIBUTING, THIRD_PARTY_NOTICES, CI, release workflow, `scripts/build.ps1`.
**Done when:** tagging `v0.1.0` produces a downloadable zip that runs on a clean Windows machine with only VB-Cable installed.

### M5 (v2): AI voice mode (NVIDIA GPU, optional)
Real-time voice conversion using an RVC-style model with PyTorch + CUDA (or ONNX Runtime with the CUDA provider), in `src/timbrel/ai/`.
- Optional install: `pip install timbrel[ai]`; the base app must work without it.
- Separate larger block size and its own latency readout; expect roughly 100–300 ms added latency depending on GPU and settings. Show this clearly in the UI.
- Users load their own model files (`.pth` / `.onnx` + index). Ship no models of real people. Add a README note about consent and not impersonating real people.
- Auto-detect CUDA; if unavailable, disable AI mode with a clear message.
**Done when:** a user-supplied model runs in real time on an RTX-class GPU without underruns.

## Conventions

- Format and lint with `ruff`; type hints everywhere; `pytest` for tests.
- Keep Windows-specific code inside `platform/windows.py`.
- Prefer small, focused commits with clear messages.
- When unsure about a requirement, ask before building rather than guessing.
