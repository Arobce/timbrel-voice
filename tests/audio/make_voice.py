"""Generate tests/audio/voice.wav: a synthetic, voice-like test clip (CC0).

Two "vowels" (harmonic series with vibrato and formant resonances) separated
by pauses with a quiet noise floor, so gate, pitch and filter effects all have
something realistic to chew on. Deterministic; rerun to regenerate.

    python tests/audio/make_voice.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import wavfile

SAMPLE_RATE = 48_000
OUT = Path(__file__).with_name("voice.wav")
# (centre Hz, bandwidth Hz) for an "ah"-like vowel
FORMANTS = [(700.0, 110.0), (1220.0, 120.0), (2600.0, 160.0)]


def vowel(duration: float, f0_start: float, f0_end: float) -> np.ndarray:
    n = int(duration * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE
    f0 = np.linspace(f0_start, f0_end, n) * (1 + 0.02 * np.sin(2 * np.pi * 5.5 * t))
    phase = 2 * np.pi * np.cumsum(f0) / SAMPLE_RATE
    out = np.zeros(n)
    for k in range(1, 40):
        freq = k * f0.mean()
        if freq > SAMPLE_RATE / 2 - 1000:
            break
        amp = sum(1.0 / (1.0 + ((freq - fc) / bw) ** 2) for fc, bw in FORMANTS) / k**0.5
        out += amp * np.sin(k * phase)
    envelope = np.minimum(1.0, np.minimum(t, t[-1] - t) / 0.03)  # 30 ms fades
    return out * envelope


def main() -> None:
    rng = np.random.default_rng(1234)

    def pause(seconds: float) -> np.ndarray:
        return np.zeros(int(seconds * SAMPLE_RATE))

    clip = np.concatenate(
        [pause(0.3), vowel(0.6, 140, 150), pause(0.25), vowel(0.6, 190, 120), pause(0.3)]
    )
    clip *= 0.7 / np.abs(clip).max()
    clip += rng.normal(0, 10 ** (-65 / 20), clip.size)  # -65 dBFS room noise
    wavfile.write(OUT, SAMPLE_RATE, (clip * 32767).astype(np.int16))
    print(f"wrote {OUT} ({clip.size / SAMPLE_RATE:.2f} s)")


if __name__ == "__main__":
    main()
