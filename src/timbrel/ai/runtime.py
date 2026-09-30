"""RVC v2 voice conversion on ONNX Runtime.

Pipeline (as in RVC's own inference):

    16 kHz audio -> HuBERT/ContentVec features (50 fps, 768) -> x2 -> 100 fps
    16 kHz audio -> log-mel (128 bands, hop 160) -> RMVPE -> pitch (100 fps)
    features (+ index retrieval) + pitch (+ shift) + noise -> generator -> 40 kHz

The DSP helpers here are plain numpy so they can be tested without
onnxruntime; the model classes import onnxruntime and faiss lazily.
"""

from __future__ import annotations

import math
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

SR16 = 16000
HOP16 = 160  # pitch/feature frames: 100 per second
OUT_SR = 40000  # RVC v2 "40k" voices
OUT_PER_FRAME = OUT_SR // 100  # 400 output samples per frame
HUBERT_SAMPLES = 32000  # the exported HuBERT takes exactly 2 s
GEN_FRAMES = 200  # voices are exported for exactly 200 frames (2 s)
NOISE_SCALE = 0.66666  # generator noise, as in RVC's own inference

HUBERT_FILE = "hubert_base_layer12_nomask_32000.onnx"
RMVPE_FILE = "rmvpe.onnx"


# --- pitch front end (numpy only) ----------------------------------------------


def mel_filterbank(
    sr: int = SR16, n_fft: int = 1024, n_mels: int = 128, fmin: float = 30.0, fmax: float = 8000.0
) -> np.ndarray:
    """HTK-scale triangular mel filters with Slaney area normalisation
    (librosa's ``mel(htk=True)``, which RMVPE was trained with)."""

    def hz_to_mel(f: np.ndarray) -> np.ndarray:
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel_to_hz(m: np.ndarray) -> np.ndarray:
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    fft_freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
    edges = mel_to_hz(np.linspace(hz_to_mel(np.array(fmin)), hz_to_mel(np.array(fmax)), n_mels + 2))
    ramps = edges[:, None] - fft_freqs[None, :]
    lower = -ramps[:-2] / np.diff(edges)[:-1, None]
    upper = ramps[2:] / np.diff(edges)[1:, None]
    weights = np.maximum(0, np.minimum(lower, upper))
    weights *= (2.0 / (edges[2 : n_mels + 2] - edges[:n_mels]))[:, None]
    return weights.astype(np.float32)


_MEL = mel_filterbank()
_WINDOW = np.hanning(1025)[:-1].astype(np.float32)  # periodic Hann, like torch.hann_window
_CENTS = 20 * np.arange(360) + 1997.3794084376191  # RMVPE's 360 pitch bins


def log_mel(audio16: np.ndarray) -> np.ndarray:
    """RMVPE's input: centred STFT (1024, hop 160) -> mel -> log(clamp 1e-5)."""
    x = np.pad(audio16, (512, 512), mode="reflect")
    n_frames = 1 + (len(x) - 1024) // HOP16
    idx = np.arange(1024)[None, :] + HOP16 * np.arange(n_frames)[:, None]
    mag = np.abs(np.fft.rfft(x[idx] * _WINDOW, n=1024, axis=1)).astype(np.float32)
    return np.log(np.clip(_MEL @ mag.T, 1e-5, None)).astype(np.float32)


def decode_f0(salience: np.ndarray, threshold: float = 0.03) -> np.ndarray:
    """RMVPE output (frames x 360) -> Hz, 0 where unvoiced. Weighted average of
    the 9 bins around the peak, as in RMVPE's own decoder."""
    center = np.argmax(salience, axis=1)
    padded = np.pad(salience, ((0, 0), (4, 4)))
    cols = center[:, None] + np.arange(9)[None, :]
    weights = padded[np.arange(len(salience))[:, None], cols]
    cents = (weights * np.pad(_CENTS, 4)[cols]).sum(1) / np.maximum(weights.sum(1), 1e-9)
    cents[salience.max(axis=1) <= threshold] = 0
    f0 = 10 * 2 ** (cents / 1200)
    f0[f0 == 10] = 0
    return f0.astype(np.float32)


def smooth_f0(f0: np.ndarray, size: int = 5) -> np.ndarray:
    """Median-filter voicing and log-pitch so single flipped frames and octave
    jumps don't reach the generator."""
    from scipy.ndimage import median_filter

    voiced = f0 > 0
    if size <= 1 or not voiced.any():
        return f0
    keep = median_filter(voiced.astype(np.float32), size, mode="nearest") > 0.5
    logf = np.log2(np.maximum(f0, 1.0))
    filled = np.where(
        voiced, logf, np.interp(np.arange(len(f0)), np.flatnonzero(voiced), logf[voiced])
    )
    smooth = 2 ** median_filter(filled, size, mode="nearest")
    return np.where(keep, smooth, 0).astype(np.float32)


def coarse_pitch(f0: np.ndarray) -> np.ndarray:
    """Hz -> RVC's 1..255 mel-scale pitch bins (1 = unvoiced)."""
    mel_min = 1127 * math.log(1 + 50 / 700)
    mel_max = 1127 * math.log(1 + 1100 / 700)
    f0_mel = 1127 * np.log(1 + f0 / 700)
    voiced = f0_mel > 0
    f0_mel[voiced] = (f0_mel[voiced] - mel_min) * 254 / (mel_max - mel_min) + 1
    return np.rint(np.clip(f0_mel, 1, 255)).astype(np.int64)


# --- models (onnxruntime / faiss imported lazily) -------------------------------


class AiUnavailable(RuntimeError):
    """The optional AI install, a CUDA GPU, or a model file is missing."""


def check_runtime() -> str:
    """Raise AiUnavailable with a user-facing reason unless AI voice can run.
    Returns a short description of the GPU provider on success."""
    try:
        import onnxruntime as ort
    except ImportError:
        raise AiUnavailable('AI voice needs the optional install: pip install -e ".[ai]"') from None
    try:
        ort.preload_dlls()
    except Exception:  # noqa: BLE001 - older builds / system CUDA installs
        pass
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        raise AiUnavailable("AI voice needs an NVIDIA GPU with CUDA; none was found.")
    return f"ONNX Runtime {ort.__version__} (CUDA)"


def _session(path: Path) -> Any:
    import onnxruntime as ort

    if not path.exists():
        raise AiUnavailable(f"Missing model file: {path}")
    options = ort.SessionOptions()
    options.log_severity_level = 3
    session = ort.InferenceSession(
        str(path), options, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    if session.get_providers()[0] != "CUDAExecutionProvider":
        raise AiUnavailable(f"{path.name} couldn't be loaded on the GPU.")
    return session


class BaseModels:
    """HuBERT (content) and RMVPE (pitch); shared by all voices."""

    _cache: dict[Path, BaseModels] = {}
    _lock = threading.Lock()

    def __init__(self, folder: Path) -> None:
        self.hubert = _session(folder / HUBERT_FILE)
        self.rmvpe = _session(folder / RMVPE_FILE)

    @classmethod
    def load(cls, folder: Path) -> BaseModels:
        """Load once per folder (a few seconds) and reuse."""
        with cls._lock:
            if folder not in cls._cache:
                cls._cache[folder] = cls(folder)
            return cls._cache[folder]


class VoiceModel:
    """One RVC v2 40 kHz voice: generator .onnx plus optional .index."""

    def __init__(self, onnx_path: Path, index_path: Path | None = None) -> None:
        self.name = onnx_path.stem
        self.generator = _session(onnx_path)
        self.index: Any = None
        self.index_vectors: np.ndarray | None = None
        if index_path is not None and index_path.exists():
            try:
                import faiss  # binary index format, not a pickle
            except ImportError:
                raise AiUnavailable(
                    'Voice index files need faiss: pip install -e ".[ai]"'
                ) from None
            self.index = faiss.read_index(str(index_path))
            self.index_vectors = self.index.reconstruct_n(0, self.index.ntotal)


class RvcConverter:
    """Converts a 2 s window of 16 kHz audio to 40 kHz in the target voice.
    Runs on the AI worker thread only."""

    def __init__(self, base: BaseModels, voice: VoiceModel, seed: int = 0) -> None:
        self.base = base
        self.voice = voice
        self.rng = np.random.default_rng(seed)
        self.index_rate = 0.5
        self.protect = 0.33  # unvoiced frames keep more of the original features
        self.f0_median = 5
        self.timings: dict[str, float] = {}

    def _time(self, key: str, start: float) -> None:
        self.timings[key] = (time.perf_counter() - start) * 1000

    def features(self, audio16: np.ndarray) -> np.ndarray:
        t = time.perf_counter()
        seg = np.zeros(HUBERT_SAMPLES, np.float32)
        seg[: min(len(audio16), HUBERT_SAMPLES)] = audio16[:HUBERT_SAMPLES]
        feats = self.base.hubert.run(None, {"source": seg[None]})[0][0]
        self._time("hubert", t)
        return feats[: len(audio16) // 320]

    def pitch(self, audio16: np.ndarray) -> np.ndarray:
        t = time.perf_counter()
        mel = log_mel(audio16)
        n = mel.shape[1]
        mel = np.pad(mel, ((0, 0), (0, (32 - n % 32) % 32)))  # U-Net wants multiples of 32
        salience = self.base.rmvpe.run(None, {"input": mel[None]})[0][0][:n]
        f0 = smooth_f0(decode_f0(salience), self.f0_median)
        self._time("rmvpe", t)
        return f0

    def retrieve(self, feats: np.ndarray) -> np.ndarray:
        """Blend each feature with its nearest neighbours in the voice's index
        (the target's own speech). Higher rates sound more like the target
        but can change words."""
        index, vectors = self.voice.index, self.voice.index_vectors
        if index is None or vectors is None or self.index_rate <= 0:
            return feats
        t = time.perf_counter()
        dist, ix = index.search(np.ascontiguousarray(feats, dtype=np.float32), 8)
        weight = np.square(1 / np.maximum(dist, 1e-8))
        weight /= weight.sum(axis=1, keepdims=True)
        matched = np.sum(vectors[ix] * weight[:, :, None], axis=1)
        self._time("index", t)
        return (matched * self.index_rate + feats * (1 - self.index_rate)).astype(np.float32)

    def convert(self, audio16: np.ndarray, semitones: float) -> np.ndarray:
        raw = self.features(audio16)
        feats = self.retrieve(raw)
        feats, raw = np.repeat(feats, 2, axis=0), np.repeat(raw, 2, axis=0)  # 50 -> 100 fps
        f0 = self.pitch(audio16)
        n = min(len(feats), len(f0), GEN_FRAMES)
        feats, raw, f0 = feats[:n], raw[:n], f0[:n] * 2 ** (semitones / 12)
        if self.voice.index is not None and self.protect < 0.5:
            keep = np.where(f0 > 0, 1.0, self.protect)[:, None].astype(np.float32)
            feats = feats * keep + raw * (1 - keep)
        pad = GEN_FRAMES - n  # the exported generator has a fixed length
        feats = np.pad(feats, ((0, pad), (0, 0)), mode="edge")
        f0 = np.pad(f0, (0, pad), mode="edge")
        t = time.perf_counter()
        noise = self.rng.standard_normal((1, 192, GEN_FRAMES)) * NOISE_SCALE
        out = self.voice.generator.run(
            None,
            {
                "phone": feats[None].astype(np.float32),
                "phone_lengths": np.array([GEN_FRAMES], dtype=np.int64),
                "pitch": coarse_pitch(f0.copy())[None],
                "pitchf": f0[None].astype(np.float32),
                "ds": np.array([0], dtype=np.int64),
                "rnd": noise.astype(np.float32),
            },
        )[0]
        self._time("generator", t)
        return out.reshape(-1)[: n * OUT_PER_FRAME]
