from __future__ import annotations

import io
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from faster_whisper import WhisperModel

DEFAULT_MODEL = "medium"
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
ALLOWED_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".m4a",
    ".mp4",
    ".aac",
    ".ogg",
    ".flac",
    ".webm",
}
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_AUDIO_DIR = Path("/audios")
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output"
SUPPORTED_DEVICES = {"cpu", "cuda", "rocm"}


def _read_env_setting(name: str) -> Optional[str]:
    value = os.getenv(name)
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


def normalize_device(device: str) -> str:
    normalized = device.strip().lower()
    if normalized in SUPPORTED_DEVICES:
        return normalized
    raise ValueError(
        f"Unsupported WHISPER_DEVICE '{device}'. Supported values: {', '.join(sorted(SUPPORTED_DEVICES))}."
    )


def get_runtime_device(device: str) -> str:
    normalized = normalize_device(device)
    if normalized == "rocm":
        # ROCm-enabled CTranslate2 builds expose the GPU backend through "cuda".
        return "cuda"
    return normalized


def apply_rocm_env_overrides() -> None:
    # Optional helper envs for legacy/unsupported ROCm cards (for example Vega).
    hsa_override = _read_env_setting("WHISPER_HSA_OVERRIDE_GFX_VERSION")
    if hsa_override and not _read_env_setting("HSA_OVERRIDE_GFX_VERSION"):
        os.environ["HSA_OVERRIDE_GFX_VERSION"] = hsa_override

    rocr_visible = _read_env_setting("WHISPER_ROCR_VISIBLE_DEVICES")
    if rocr_visible and not _read_env_setting("ROCR_VISIBLE_DEVICES"):
        os.environ["ROCR_VISIBLE_DEVICES"] = rocr_visible


def is_cpu_device(device: str) -> bool:
    return normalize_device(device) == "cpu"


def get_default_device() -> str:
    return normalize_device(_read_env_setting("WHISPER_DEVICE") or "cuda")


def get_default_compute_type(device: str) -> str:
    configured = _read_env_setting("WHISPER_COMPUTE_TYPE")
    if configured:
        return configured.lower()
    if is_cpu_device(device):
        return "int8"
    return "float16"


DEFAULT_DEVICE = get_default_device()
DEFAULT_COMPUTE_TYPE = get_default_compute_type(DEFAULT_DEVICE)


class TranscriptionError(Exception):
    pass


class UploadValidationError(TranscriptionError):
    pass


@dataclass
class SegmentResult:
    index: int
    start: float
    end: float
    text: str


@dataclass
class TranscriptionInfo:
    duration: Optional[float]


@dataclass
class AudioLibraryFile:
    filename: str
    path: str
    size_bytes: int


class ModelManager:
    def __init__(self, device: Optional[str] = None, compute_type: Optional[str] = None) -> None:
        resolved_device = normalize_device(device or DEFAULT_DEVICE)
        if resolved_device == "rocm":
            apply_rocm_env_overrides()
        resolved_compute_type = (
            compute_type.strip().lower() if compute_type and compute_type.strip() else get_default_compute_type(resolved_device)
        )
        self.device = resolved_device
        self.runtime_device = get_runtime_device(resolved_device)
        self.compute_type = resolved_compute_type
        self._models: dict[str, WhisperModel] = {}
        self._lock = threading.Lock()

    def get(self, model_name: str) -> WhisperModel:
        with self._lock:
            if model_name in self._models:
                return self._models[model_name]

            try:
                model = WhisperModel(model_name, device=self.runtime_device, compute_type=self.compute_type)
            except Exception as exc:  # noqa: BLE001
                runtime_hint = (
                    f"{self.device} (mapped runtime backend: {self.runtime_device})"
                    if self.device != self.runtime_device
                    else self.device
                )
                raise TranscriptionError(
                    f"Failed to initialize model '{model_name}' with device={runtime_hint}, "
                    f"compute_type={self.compute_type}. For WHISPER_DEVICE=rocm, install a ROCm-enabled "
                    "CTranslate2 build and pass ROCm devices into the container/host runtime. "
                    f"Original error: {exc}"
                ) from exc

            self._models[model_name] = model
            return model


def validate_upload(filename: str, payload: bytes) -> None:
    if not payload:
        raise UploadValidationError("Uploaded file is empty.")
    if len(payload) > MAX_UPLOAD_BYTES:
        raise UploadValidationError("Uploaded file exceeds the 50 MB limit.")

    lowered = filename.lower()
    if not any(lowered.endswith(ext) for ext in ALLOWED_EXTENSIONS):
        raise UploadValidationError(
            "Unsupported file extension. Allowed: " + ", ".join(sorted(ALLOWED_EXTENSIONS))
        )


def get_audio_library_dir() -> Path:
    configured = os.getenv("AUDIO_DIR")
    if configured:
        candidate = Path(configured).expanduser()
        if not candidate.is_absolute():
            candidate = PROJECT_ROOT / candidate
        return candidate

    local_default = PROJECT_ROOT / "audios"
    if local_default.exists():
        return local_default
    return DEFAULT_AUDIO_DIR


def get_output_dir() -> Path:
    configured = os.getenv("OUTPUT_DIR")
    if configured:
        candidate = Path(configured).expanduser()
        if not candidate.is_absolute():
            candidate = PROJECT_ROOT / candidate
        return candidate
    return DEFAULT_OUTPUT_DIR


def list_audio_library_files(audio_dir: Optional[Path] = None) -> list[AudioLibraryFile]:
    directory = audio_dir or get_audio_library_dir()
    if not directory.exists() or not directory.is_dir():
        return []

    files: list[AudioLibraryFile] = []
    for candidate in directory.rglob("*"):
        if not candidate.is_file():
            continue

        lowered = candidate.name.lower()
        if not any(lowered.endswith(ext) for ext in ALLOWED_EXTENSIONS):
            continue

        relative = candidate.relative_to(directory).as_posix()
        files.append(
            AudioLibraryFile(
                filename=candidate.name,
                path=relative,
                size_bytes=candidate.stat().st_size,
            )
        )

    files.sort(key=lambda item: item.path.lower())
    return files


def load_audio_from_library(audio_path: str, audio_dir: Optional[Path] = None) -> tuple[str, bytes]:
    normalized_path = (audio_path or "").strip()
    if not normalized_path:
        raise UploadValidationError("Missing audio path.")

    directory = (audio_dir or get_audio_library_dir()).resolve()
    if not directory.exists() or not directory.is_dir():
        raise UploadValidationError(f"Audio directory not found: {directory}")

    candidate = (directory / normalized_path).resolve()
    try:
        candidate.relative_to(directory)
    except ValueError as exc:
        raise UploadValidationError("Invalid audio path outside /audios directory.") from exc

    if not candidate.is_file():
        raise UploadValidationError(f"Audio file not found in /audios: {normalized_path}")

    payload = candidate.read_bytes()
    validate_upload(candidate.name, payload)
    return candidate.name, payload


def transcribe_audio(
    model_manager: ModelManager,
    audio_bytes: bytes,
    model_name: str,
    language: Optional[str],
):
    model = model_manager.get(model_name)
    audio_stream = io.BytesIO(audio_bytes)

    whisper_language = None if language in (None, "", "auto") else language

    try:
        segments, info = model.transcribe(
            audio_stream,
            language=whisper_language,
            vad_filter=True,
            condition_on_previous_text=True,
        )
    except Exception as exc:  # noqa: BLE001
        raise TranscriptionError(f"Failed to decode or transcribe audio: {exc}") from exc

    duration = getattr(info, "duration", None)
    yield TranscriptionInfo(duration=duration)

    for idx, segment in enumerate(segments):
        yield SegmentResult(
            index=idx,
            start=float(getattr(segment, "start", 0.0)),
            end=float(getattr(segment, "end", 0.0)),
            text=(getattr(segment, "text", "") or "").strip(),
        )
