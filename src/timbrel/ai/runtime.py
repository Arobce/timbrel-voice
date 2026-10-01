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
GEN_FRAMES = 200  # exported voices only run at exactly 200 frames (2 s)
PITCH_TAIL = 96  # frames RMVPE sees per streaming step (a multiple of 32)
PITCH_REDO = 64  # newest frames recomputed per step; older ones are reused
NOISE_SCALE = 0.66666  # generator noise, as in RVC's own inference

HUBERT_FRAMES = 99  # its output frames (50 fps)
LOG_FLOOR = math.log(1e-5)  # log-mel of silence

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
    _load_cuda_dlls()
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        raise AiUnavailable("AI voice needs an NVIDIA GPU with CUDA; none was found.")
    return f"ONNX Runtime {ort.__version__} (CUDA)"


_dlls_loaded = False


def _load_cuda_dlls() -> None:
    """Load NVIDIA's CUDA/cuDNN DLLs from their pip wheels (once)."""
    global _dlls_loaded
    if _dlls_loaded:
        return
    import onnxruntime as ort

    try:
        ort.preload_dlls()
    except Exception:  # noqa: BLE001 - older builds / system CUDA installs
        pass
    _dlls_loaded = True


def _session(path: Path, cuda_graph: bool = False) -> Any:
    import onnxruntime as ort

    _load_cuda_dlls()
    if not path.exists():
        raise AiUnavailable(f"Missing model file: {path}")
    options = ort.SessionOptions()
    options.log_severity_level = 3
    cuda = ("CUDAExecutionProvider", {"enable_cuda_graph": "1"} if cuda_graph else {})
    providers: list[Any] = [cuda] if cuda_graph else [cuda, "CPUExecutionProvider"]
    session = ort.InferenceSession(str(path), options, providers=providers)
    if session.get_providers()[0] != "CUDAExecutionProvider":
        raise AiUnavailable(f"{path.name} couldn't be loaded on the GPU.")
    return session


class FixedRunner:
    """Runs a model at fixed input shapes. The GPU work is recorded once as a
    CUDA graph and replayed, which roughly halves the time of these small,
    many-layer models (kernel launches dominate). Falls back to ordinary runs
    if the model can't be captured. Thread-safe: voices share base models."""

    def __init__(
        self, path: Path, inputs: dict[str, np.ndarray], out_shape: tuple[int, ...]
    ) -> None:
        self._lock = threading.Lock()
        self._dtypes = {k: a.dtype for k, a in inputs.items()}
        try:
            import onnxruntime as ort

            self.session = _session(path, cuda_graph=True)
            self._inputs = {
                k: ort.OrtValue.ortvalue_from_numpy(a, "cuda", 0) for k, a in inputs.items()
            }
            self._output = ort.OrtValue.ortvalue_from_numpy(
                np.zeros(out_shape, np.float32), "cuda", 0
            )
            self._binding = self.session.io_binding()
            for k, value in self._inputs.items():
                self._binding.bind_ortvalue_input(k, value)
            self._binding.bind_ortvalue_output(self.session.get_outputs()[0].name, self._output)
            self.session.run_with_iobinding(self._binding)  # records the graph
            self.graph = True
        except AiUnavailable:
            raise
        except Exception:  # noqa: BLE001 - some models/drivers can't be captured
            self.session = _session(path)
            self.graph = False

    def run(self, feeds: dict[str, np.ndarray]) -> np.ndarray:
        """First output for ``feeds`` (same shapes as at construction)."""
        with self._lock:
            if not self.graph:
                return self.session.run(None, feeds)[0]
            for k, a in feeds.items():
                self._inputs[k].update_inplace(np.ascontiguousarray(a, dtype=self._dtypes[k]))
            self.session.run_with_iobinding(self._binding)
            return self._output.numpy()


class BaseModels:
    """HuBERT (content) and RMVPE (pitch); shared by all voices."""

    _cache: dict[Path, BaseModels] = {}
    _lock = threading.Lock()

    def __init__(self, folder: Path) -> None:
        self.hubert = FixedRunner(
            folder / HUBERT_FILE,
            {"source": np.zeros((1, HUBERT_SAMPLES), np.float32)},
            (1, HUBERT_FRAMES, 768),
        )
        self.rmvpe = FixedRunner(
            folder / RMVPE_FILE,
            {"input": np.full((1, 128, PITCH_TAIL), LOG_FLOOR, np.float32)},
            (1, PITCH_TAIL, 360),
        )

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
        f = GEN_FRAMES
        self.generator = FixedRunner(
            onnx_path,
            {
                "phone": np.zeros((1, f, 768), np.float32),
                "phone_lengths": np.array([f], np.int64),
                "pitch": np.ones((1, f), np.int64),
                "pitchf": np.zeros((1, f), np.float32),
                "ds": np.array([0], np.int64),
                "rnd": np.zeros((1, 192, f), np.float32),
            },
            (1, 1, f * OUT_PER_FRAME),
        )
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
        self._salience: np.ndarray | None = None  # RMVPE output for the last window

    def _time(self, key: str, start: float) -> None:
        self.timings[key] = (time.perf_counter() - start) * 1000

    def features(self, audio16: np.ndarray) -> np.ndarray:
        t = time.perf_counter()
        seg = np.zeros(HUBERT_SAMPLES, np.float32)
        seg[: min(len(audio16), HUBERT_SAMPLES)] = audio16[:HUBERT_SAMPLES]
        feats = self.base.hubert.run({"source": seg[None]})[0]
        self._time("hubert", t)
        return feats[: len(audio16) // 320]

    def pitch(self, audio16: np.ndarray, advance16: int | None = None) -> np.ndarray:
        """Pitch per 10 ms frame. With ``advance16`` (samples the window moved
        since the last call) only the newest frames are recomputed: older
        ones already had plenty of context on both sides."""
        t = time.perf_counter()
        mel = log_mel(audio16)
        n = mel.shape[1]
        prev = self._salience
        shift = None if advance16 is None or advance16 % HOP16 else advance16 // HOP16
        if prev is not None and shift is not None and shift <= PITCH_REDO and len(prev) == n:
            salience = np.empty_like(prev)
            salience[: n - shift] = prev[shift:]
            self._salience_to(salience, mel, n)
        else:  # from scratch, newest frames first, PITCH_REDO at a time
            salience = np.empty((n, 360), np.float32)
            for end in range(n, 0, -PITCH_REDO):
                self._salience_to(salience, mel, end)
        self._salience = salience
        f0 = smooth_f0(decode_f0(salience), self.f0_median)
        self._time("rmvpe", t)
        return f0

    def _salience_to(self, salience: np.ndarray, mel: np.ndarray, end: int) -> None:
        """Fill ``salience[end - PITCH_REDO : end]`` from one RMVPE run over
        the PITCH_TAIL frames ending at ``end`` (the extra frames are context)."""
        start = max(0, end - PITCH_TAIL)
        window = np.full((1, 128, PITCH_TAIL), LOG_FLOOR, np.float32)
        seg = mel[:, start : start + PITCH_TAIL]
        window[0, :, : seg.shape[1]] = seg
        out = self.base.rmvpe.run({"input": window})[0]
        lo = max(0, end - PITCH_REDO)
        salience[lo:end] = out[lo - start : end - start]

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

    def convert(
        self, audio16: np.ndarray, semitones: float, advance16: int | None = None
    ) -> np.ndarray:
        """``advance16``: how far the window moved since the previous call
        (None if unknown, e.g. after a reset)."""
        raw = self.features(audio16)
        feats = self.retrieve(raw)
        feats, raw = np.repeat(feats, 2, axis=0), np.repeat(raw, 2, axis=0)  # 50 -> 100 fps
        f0 = self.pitch(audio16, advance16)
        # The generator sees the newest GEN_FRAMES; its output ends where the
        # window's features end.
        frames = GEN_FRAMES
        end = min(len(feats), len(f0))
        start = max(0, end - frames)
        n = end - start
        feats, raw = feats[start:end], raw[start:end]
        f0 = f0[start:end] * 2 ** (semitones / 12)
        if self.voice.index is not None and self.protect < 0.5:
            keep = np.where(f0 > 0, 1.0, self.protect)[:, None].astype(np.float32)
            feats = feats * keep + raw * (1 - keep)
        pad = frames - n  # the generator runs at a fixed length
        feats = np.pad(feats, ((0, pad), (0, 0)), mode="edge")
        f0 = np.pad(f0, (0, pad), mode="edge")
        t = time.perf_counter()
        noise = self.rng.standard_normal((1, 192, frames)) * NOISE_SCALE
        out = self.voice.generator.run(
            {
                "phone": feats[None].astype(np.float32),
                "phone_lengths": np.array([frames], dtype=np.int64),
                "pitch": coarse_pitch(f0.copy())[None],
                "pitchf": f0[None].astype(np.float32),
                "ds": np.array([0], dtype=np.int64),
                "rnd": noise.astype(np.float32),
            },
        )
        self._time("generator", t)
        return out.reshape(-1)[: n * OUT_PER_FRAME]
