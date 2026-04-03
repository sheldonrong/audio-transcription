from __future__ import annotations

import io
import json
import os
import re
import subprocess
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

from faster_whisper import WhisperModel

DEFAULT_MODEL = "medium"
MAX_UPLOAD_BYTES = 1024 * 1024 * 1024
MAX_UPLOAD_MB = MAX_UPLOAD_BYTES // (1024 * 1024)
ALLOWED_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".m4a",
    ".mp4",
    ".aac",
    ".ogg",
    ".flac",
    ".webm",
    ".3gp",
    ".amr",
    ".opus",
    ".caf",
}
VIDEO_ALLOWED_EXTENSIONS = {
    ".mkv",
    ".mp4",
}
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_AUDIO_DIR = Path("/audios")
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output"
SUPPORTED_DEVICES = {"cpu", "cuda", "rocm"}


def get_upload_limit_message() -> str:
    return f"Uploaded file exceeds the {MAX_UPLOAD_MB} MB limit."


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


@dataclass(frozen=True)
class AmdGpuInfo:
    device_id: int
    name: str
    gfx_version: Optional[str] = None


@dataclass(frozen=True)
class AmdGpuInventory:
    gpus: list[AmdGpuInfo]
    detected_at: str
    detection_method: Optional[str] = None
    detection_error: Optional[str] = None


def _clean_optional_text(value: object) -> Optional[str]:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def _extract_card_index(value: object) -> Optional[int]:
    if not isinstance(value, str):
        return None
    match = re.search(r"(\d+)", value)
    if not match:
        return None
    return int(match.group(1))


def _parse_rocm_smi_gpus(payload: str) -> list[AmdGpuInfo]:
    try:
        raw = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid rocm-smi JSON output: {exc}") from exc

    if not isinstance(raw, dict):
        raise ValueError("Unexpected rocm-smi output shape.")

    gpus: list[AmdGpuInfo] = []
    for card_name, details in raw.items():
        if not isinstance(details, dict):
            continue

        device_id = _extract_card_index(card_name)
        if device_id is None:
            continue

        name = _clean_optional_text(details.get("Card Series"))
        gfx_version = _clean_optional_text(details.get("GFX Version"))
        gpus.append(AmdGpuInfo(device_id=device_id, name=name, gfx_version=gfx_version))

    if not gpus:
        raise ValueError("No GPU entries found in rocm-smi output.")

    return sorted(gpus, key=lambda gpu: gpu.device_id)


def _parse_rocminfo_gpus(payload: str) -> list[AmdGpuInfo]:
    blocks = re.split(r"(?m)^Agent\s+\d+\s*$", payload)
    gpus: list[AmdGpuInfo] = []

    for block in blocks:
        lines = [line.rstrip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue

        parsed: dict[str, str] = {}
        for line in lines:
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            parsed[key.strip()] = value.strip()

        device_type = parsed.get("Type", "").strip().upper()
        if device_type != "GPU":
            continue

        name = parsed.get("Marketing Name") or parsed.get("Name") or f"AMD GPU {len(gpus)}"
        gfx_version = parsed.get("GFXIP")
        gpus.append(
            AmdGpuInfo(
                device_id=len(gpus),
                name=name.strip(),
                gfx_version=gfx_version.strip() if isinstance(gfx_version, str) and gfx_version.strip() else None,
            )
        )

    if not gpus:
        raise ValueError("No GPU agents found in rocminfo output.")

    return gpus


def detect_amd_gpus() -> AmdGpuInventory:
    detected_at = datetime.now(timezone.utc).isoformat()
    detection_attempts = (
        (
            "rocm-smi",
            ["rocm-smi", "--showproductname", "--showgfxversion", "--json"],
            _parse_rocm_smi_gpus,
        ),
        ("rocminfo", ["rocminfo"], _parse_rocminfo_gpus),
    )
    errors: list[str] = []

    for method_name, command, parser in detection_attempts:
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=True)
        except FileNotFoundError:
            errors.append(f"{method_name} not found")
            continue
        except subprocess.CalledProcessError as exc:
            detail = exc.stderr.strip() or exc.stdout.strip() or str(exc)
            errors.append(f"{method_name} failed: {detail}")
            continue

        try:
            gpus = parser(result.stdout)
        except ValueError as exc:
            errors.append(f"{method_name} parse error: {exc}")
            continue

        return AmdGpuInventory(
            gpus=gpus,
            detected_at=detected_at,
            detection_method=method_name,
        )

    detection_error = "; ".join(errors) if errors else "No AMD GPU detection method succeeded."
    return AmdGpuInventory(gpus=[], detected_at=detected_at, detection_error=detection_error)


def get_detected_amd_gpu_inventory() -> AmdGpuInventory:
    return DETECTED_AMD_GPU_INVENTORY


def normalize_device_ids(device_ids: Optional[Sequence[int]]) -> tuple[int, ...]:
    if not device_ids:
        return ()

    normalized: list[int] = []
    seen: set[int] = set()
    for raw_value in device_ids:
        value = int(raw_value)
        if value < 0:
            raise ValueError("GPU device IDs must be zero or greater.")
        if value in seen:
            continue
        seen.add(value)
        normalized.append(value)

    return tuple(normalized)


def validate_device_ids(device: str, device_ids: Optional[Sequence[int]]) -> tuple[int, ...]:
    normalized = normalize_device_ids(device_ids)
    if not normalized:
        return normalized

    if is_cpu_device(device):
        raise TranscriptionError("GPU device IDs are not supported when WHISPER_DEVICE=cpu.")

    detected = get_detected_amd_gpu_inventory()
    if detected.gpus:
        available_ids = {gpu.device_id for gpu in detected.gpus}
        invalid_ids = [device_id for device_id in normalized if device_id not in available_ids]
        if invalid_ids:
            raise TranscriptionError(
                "Unknown AMD GPU device ID(s): "
                + ", ".join(str(device_id) for device_id in invalid_ids)
                + ". Available IDs: "
                + ", ".join(str(device_id) for device_id in sorted(available_ids))
            )

    return normalized


DETECTED_AMD_GPU_INVENTORY = detect_amd_gpus()


class ModelManager:
    def __init__(self, device: Optional[str] = None, compute_type: Optional[str] = None) -> None:
        resolved_device = normalize_device(device or DEFAULT_DEVICE)
        resolved_compute_type = (
            compute_type.strip().lower() if compute_type and compute_type.strip() else get_default_compute_type(resolved_device)
        )
        self.device = resolved_device
        self.runtime_device = get_runtime_device(resolved_device)
        self.compute_type = resolved_compute_type
        self._models: dict[tuple[str, tuple[int, ...]], WhisperModel] = {}
        self._lock = threading.Lock()

    def get(self, model_name: str, device_ids: Optional[Sequence[int]] = None) -> WhisperModel:
        normalized_device_ids = validate_device_ids(self.device, device_ids)
        cache_key = (model_name, normalized_device_ids)

        with self._lock:
            if cache_key in self._models:
                return self._models[cache_key]

            try:
                model_kwargs: dict[str, object] = {
                    "device": self.runtime_device,
                    "compute_type": self.compute_type,
                }
                if normalized_device_ids:
                    model_kwargs["device_index"] = (
                        list(normalized_device_ids)
                        if len(normalized_device_ids) > 1
                        else normalized_device_ids[0]
                    )
                    if len(normalized_device_ids) > 1:
                        model_kwargs["num_workers"] = len(normalized_device_ids)

                model = WhisperModel(model_name, **model_kwargs)
            except Exception as exc:  # noqa: BLE001
                runtime_hint = (
                    f"{self.device} (mapped runtime backend: {self.runtime_device})"
                    if self.device != self.runtime_device
                    else self.device
                )
                device_ids_hint = (
                    f", device_ids={list(normalized_device_ids)}"
                    if normalized_device_ids
                    else ""
                )
                raise TranscriptionError(
                    f"Failed to initialize model '{model_name}' with device={runtime_hint}, "
                    f"compute_type={self.compute_type}{device_ids_hint}. For WHISPER_DEVICE=rocm, install a ROCm-enabled "
                    "CTranslate2 build and pass ROCm devices into the container/host runtime. "
                    f"Original error: {exc}"
                ) from exc

            self._models[cache_key] = model
            return model


def validate_upload(filename: str, payload: bytes) -> None:
    validate_upload_size(len(payload))
    validate_upload_extension(filename)


def validate_upload_size(size_bytes: int) -> None:
    if size_bytes <= 0:
        raise UploadValidationError("Uploaded file is empty.")
    if size_bytes > MAX_UPLOAD_BYTES:
        raise UploadValidationError(get_upload_limit_message())


def validate_upload_extension(filename: str) -> None:
    lowered = filename.lower()
    if not any(lowered.endswith(ext) for ext in ALLOWED_EXTENSIONS):
        raise UploadValidationError(
            "Unsupported file extension. Allowed: " + ", ".join(sorted(ALLOWED_EXTENSIONS))
        )


def validate_video_upload_extension(filename: str) -> None:
    lowered = filename.lower()
    if not any(lowered.endswith(ext) for ext in VIDEO_ALLOWED_EXTENSIONS):
        raise UploadValidationError(
            "Unsupported video extension. Allowed: " + ", ".join(sorted(VIDEO_ALLOWED_EXTENSIONS))
        )


def _resolve_library_file(
    relative_path: str,
    *,
    audio_dir: Optional[Path],
    extension_validator,
    missing_path_message: str,
    missing_file_message: str,
    outside_path_message: str,
) -> Path:
    normalized_path = (relative_path or "").strip()
    if not normalized_path:
        raise UploadValidationError(missing_path_message)

    directory = (audio_dir or get_audio_library_dir()).resolve()
    if not directory.exists() or not directory.is_dir():
        raise UploadValidationError(f"Audio directory not found: {directory}")

    candidate = (directory / normalized_path).resolve()
    try:
        candidate.relative_to(directory)
    except ValueError as exc:
        raise UploadValidationError(outside_path_message) from exc

    extension_validator(candidate.name)

    if not candidate.is_file():
        raise UploadValidationError(f"{missing_file_message}: {normalized_path}")

    validate_upload_size(candidate.stat().st_size)
    return candidate


def resolve_audio_library_file(audio_path: str, audio_dir: Optional[Path] = None) -> Path:
    return _resolve_library_file(
        audio_path,
        audio_dir=audio_dir,
        extension_validator=validate_upload_extension,
        missing_path_message="Missing audio path.",
        missing_file_message="Audio file not found in /audios",
        outside_path_message="Invalid audio path outside /audios directory.",
    )


def resolve_video_library_file(video_path: str, audio_dir: Optional[Path] = None) -> Path:
    return _resolve_library_file(
        video_path,
        audio_dir=audio_dir,
        extension_validator=validate_video_upload_extension,
        missing_path_message="Missing video path.",
        missing_file_message="Video file not found in /audios",
        outside_path_message="Invalid video path outside /audios directory.",
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


def _list_library_files(audio_dir: Optional[Path], allowed_extensions: set[str]) -> list[AudioLibraryFile]:
    directory = audio_dir or get_audio_library_dir()
    if not directory.exists() or not directory.is_dir():
        return []

    files: list[AudioLibraryFile] = []
    for candidate in directory.rglob("*"):
        if not candidate.is_file():
            continue

        lowered = candidate.name.lower()
        if not any(lowered.endswith(ext) for ext in allowed_extensions):
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


def list_audio_library_files(audio_dir: Optional[Path] = None) -> list[AudioLibraryFile]:
    return _list_library_files(audio_dir, ALLOWED_EXTENSIONS)


def list_video_library_files(audio_dir: Optional[Path] = None) -> list[AudioLibraryFile]:
    return _list_library_files(audio_dir, VIDEO_ALLOWED_EXTENSIONS)


def load_audio_from_library(audio_path: str, audio_dir: Optional[Path] = None) -> tuple[str, bytes]:
    candidate = resolve_audio_library_file(audio_path, audio_dir)
    payload = candidate.read_bytes()
    validate_upload(candidate.name, payload)
    return candidate.name, payload


def transcribe_audio(
    model_manager: ModelManager,
    audio_source: bytes | str | Path,
    model_name: str,
    language: Optional[str],
    device_ids: Optional[Sequence[int]] = None,
    cancel_event: Optional[threading.Event] = None,
):
    model = model_manager.get(model_name, device_ids=device_ids)
    audio_input = io.BytesIO(audio_source) if isinstance(audio_source, bytes) else str(audio_source)

    whisper_language = None if language in (None, "", "auto") else language

    try:
        segments, info = model.transcribe(
            audio_input,
            language=whisper_language,
            vad_filter=True,
            condition_on_previous_text=True,
        )
    except Exception as exc:  # noqa: BLE001
        raise TranscriptionError(f"Failed to decode or transcribe audio: {exc}") from exc

    duration = getattr(info, "duration", None)
    yield TranscriptionInfo(duration=duration)

    for idx, segment in enumerate(segments):
        if cancel_event and cancel_event.is_set():
            return
        yield SegmentResult(
            index=idx,
            start=float(getattr(segment, "start", 0.0)),
            end=float(getattr(segment, "end", 0.0)),
            text=(getattr(segment, "text", "") or "").strip(),
        )
