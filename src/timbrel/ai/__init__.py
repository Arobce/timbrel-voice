"""Optional AI Voice mode (experimental): real-time RVC voice conversion.

Needs the optional install (``pip install -e ".[ai]"``) and an NVIDIA GPU.
Timbrel ships no voices: users put their own ``.onnx`` voice files (plus
optional ``.index``) in ``%APPDATA%\\Timbrel\\voices`` and the two base models
in ``%APPDATA%\\Timbrel\\ai``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from timbrel.platform import windows

# The AI voice wants a gentler gate than the classic presets: quiet syllables
# gated to silence come out unvoiced (measured: -50 dB lost ~5% more voicing).
AI_GATE = {"threshold_db": -55.0, "release_ms": 250.0}


def ai_dir() -> Path:
    """Base models (HuBERT, RMVPE)."""
    return windows.app_data_dir() / "ai"


def voices_dir() -> Path:
    return windows.app_data_dir() / "voices"


@dataclass(frozen=True)
class VoiceFile:
    name: str
    onnx: Path
    index: Path | None


def list_voices(folder: Path | None = None) -> list[VoiceFile]:
    """Voices in the folder: each ``name.onnx``, with ``name.index`` if present."""
    folder = folder or voices_dir()
    if not folder.is_dir():
        return []
    voices = []
    for onnx in sorted(folder.glob("*.onnx"), key=lambda p: p.stem.lower()):
        index = onnx.with_suffix(".index")
        voices.append(VoiceFile(onnx.stem, onnx, index if index.exists() else None))
    return voices


def load_voice(voice: VoiceFile, sample_rate: int, block_size: int, base: Path | None = None):  # noqa: ANN201
    """Build an AiVoice effect for ``voice`` (slow: loads models onto the GPU;
    call off the UI thread). Raises AiUnavailable with a user-facing reason."""
    from timbrel.ai.runtime import BaseModels, RvcConverter, VoiceModel, check_runtime
    from timbrel.ai.streaming import SAMPLE_RATE, StreamProcessor
    from timbrel.ai.voice_effect import AiVoice

    if sample_rate != SAMPLE_RATE:
        raise ValueError(f"AI voice runs at {SAMPLE_RATE} Hz")
    check_runtime()
    converter = RvcConverter(BaseModels.load(base or ai_dir()), VoiceModel(voice.onnx, voice.index))
    processor = StreamProcessor(converter)
    processor.process(processor.context[: processor.hop].copy())  # warm up the GPU
    processor.reset()
    return AiVoice(sample_rate, block_size, processor, voice.name)
