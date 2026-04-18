from __future__ import annotations

import io
import json
import logging
import os
import queue
import re
import subprocess
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from faster_whisper import WhisperModel
from faster_whisper.audio import decode_audio
from faster_whisper.transcribe import restore_speech_timestamps
from faster_whisper.vad import VadOptions, collect_chunks, get_speech_timestamps

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
logger = logging.getLogger(__name__)


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


def get_default_batch_size() -> int:
    configured = _read_env_setting("WHISPER_BATCH_SIZE")
    if not configured:
        return 2

    try:
        value = int(configured)
    except ValueError as exc:
        raise ValueError(f"Invalid WHISPER_BATCH_SIZE '{configured}'. Expected a positive integer.") from exc

    if value <= 0:
        raise ValueError(f"Invalid WHISPER_BATCH_SIZE '{configured}'. Expected a positive integer.")

    return value


DEFAULT_DEVICE = get_default_device()
DEFAULT_COMPUTE_TYPE = get_default_compute_type(DEFAULT_DEVICE)
DEFAULT_BATCH_SIZE = get_default_batch_size()


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
class AudioChunk:
    index: int
    waveform: np.ndarray
    speech_segments: list[dict[str, int]]


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

        name = (
            _clean_optional_text(details.get("Card Series"))
            or _clean_optional_text(details.get("Device Name"))
            or _clean_optional_text(details.get("Marketing Name"))
            or f"AMD GPU {device_id}"
        )
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
            "rocm-smi --showhw",
            ["rocm-smi", "--showproductname", "--showhw", "--json"],
            _parse_rocm_smi_gpus,
        ),
        (
            "rocm-smi",
            ["rocm-smi", "--showproductname", "--json"],
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
        self._lock = threading.RLock()

    def get(self, model_name: str, device_ids: Optional[Sequence[int]] = None) -> WhisperModel:
        normalized_device_ids = validate_device_ids(self.device, device_ids)
        cache_key = (model_name, normalized_device_ids)

        with self._lock:
            if cache_key in self._models:
                logger.info(
                    "WhisperModel cache hit pid=%s model=%s device_ids=%s",
                    os.getpid(),
                    model_name,
                    list(normalized_device_ids),
                )
                return self._models[cache_key]

            logger.info(
                "WhisperModel cache miss pid=%s model=%s device_ids=%s runtime_device=%s compute_type=%s",
                os.getpid(),
                model_name,
                list(normalized_device_ids),
                self.runtime_device,
                self.compute_type,
            )

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


def _build_vad_options(
    chunk_length: int,
    vad_parameters: Optional[dict[str, object] | VadOptions] = None,
) -> VadOptions:
    if vad_parameters is None:
        return VadOptions(
            max_speech_duration_s=chunk_length,
            min_silence_duration_ms=160,
        )

    values = asdict(vad_parameters) if isinstance(vad_parameters, VadOptions) else dict(vad_parameters)
    values["max_speech_duration_s"] = chunk_length
    return VadOptions(**values)


def _split_audio_with_vad(model: WhisperModel, audio: np.ndarray) -> list[AudioChunk]:
    sampling_rate = model.feature_extractor.sampling_rate
    chunk_length = model.feature_extractor.chunk_length
    vad_options = _build_vad_options(chunk_length)
    speech_segments = get_speech_timestamps(
        audio,
        vad_options=vad_options,
        sampling_rate=sampling_rate,
    )
    audio_chunks, chunks_metadata = collect_chunks(
        audio,
        speech_segments,
        sampling_rate=sampling_rate,
        max_duration=chunk_length,
    )

    chunks = [
        AudioChunk(
            index=index,
            waveform=waveform,
            speech_segments=list(metadata.get("segments", [])),
        )
        for index, (waveform, metadata) in enumerate(zip(audio_chunks, chunks_metadata))
        if waveform.size > 0
    ]

    kept_duration = (
        sum((segment["end"] - segment["start"]) for segment in speech_segments) / sampling_rate
        if speech_segments
        else 0.0
    )
    removed_duration = max((audio.shape[0] / sampling_rate) - kept_duration, 0.0)
    logger.info(
        "Manual VAD split %s chunk(s) from %s speech segment(s); removed %.2fs of audio",
        len(chunks),
        len(speech_segments),
        removed_duration,
    )
    return chunks


def _transcribe_chunk(
    model: WhisperModel,
    chunk: AudioChunk,
    language: Optional[str],
    sampling_rate: int,
) -> tuple[int, list[SegmentResult]]:
    if not chunk.speech_segments:
        return chunk.index, []

    segments, _ = model.transcribe(
        chunk.waveform,
        language=language,
        vad_filter=False,
        condition_on_previous_text=False,
    )

    restored_segments = restore_speech_timestamps(segments, chunk.speech_segments, sampling_rate)
    normalized_segments: list[SegmentResult] = []

    for raw_segment in restored_segments:
        text = (getattr(raw_segment, "text", "") or "").strip()
        if not text:
            continue
        normalized_segments.append(
            SegmentResult(
                index=-1,
                start=float(getattr(raw_segment, "start", 0.0)),
                end=float(getattr(raw_segment, "end", 0.0)),
                text=text,
            )
        )

    return chunk.index, normalized_segments


def _get_worker_models(
    model_manager: ModelManager,
    model_name: str,
    device_ids: Optional[Sequence[int]],
) -> list[WhisperModel]:
    normalized_device_ids = validate_device_ids(model_manager.device, device_ids)
    if not normalized_device_ids:
        return [model_manager.get(model_name)]
    return [model_manager.get(model_name, device_ids=[device_id]) for device_id in normalized_device_ids]


def _iter_parallel_chunk_transcriptions(
    worker_models: Sequence[WhisperModel],
    chunks: Sequence[AudioChunk],
    language: Optional[str],
    cancel_event: Optional[threading.Event],
):
    sampling_rate = worker_models[0].feature_extractor.sampling_rate
    chunk_queue: queue.Queue[AudioChunk] = queue.Queue()
    result_queue: queue.Queue[tuple[int, list[SegmentResult]] | Exception] = queue.Queue()
    stop_event = threading.Event()

    for chunk in chunks:
        chunk_queue.put(chunk)

    def worker(model: WhisperModel) -> None:
        while not stop_event.is_set() and not (cancel_event and cancel_event.is_set()):
            try:
                chunk = chunk_queue.get_nowait()
            except queue.Empty:
                return

            try:
                result_queue.put(_transcribe_chunk(model, chunk, language, sampling_rate))
            except Exception as exc:  # noqa: BLE001
                stop_event.set()
                result_queue.put(exc)
                return
            finally:
                chunk_queue.task_done()

    for model in worker_models:
        threading.Thread(target=worker, args=(model,), daemon=True).start()

    buffered_results: dict[int, list[SegmentResult]] = {}
    next_chunk_index = 0
    next_segment_index = 0
    completed_chunks = 0

    while completed_chunks < len(chunks):
        if cancel_event and cancel_event.is_set():
            stop_event.set()
            return

        try:
            item = result_queue.get(timeout=0.2)
        except queue.Empty:
            continue

        if isinstance(item, Exception):
            stop_event.set()
            raise item

        chunk_index, chunk_segments = item
        buffered_results[chunk_index] = chunk_segments
        completed_chunks += 1

        while next_chunk_index in buffered_results:
            for segment in buffered_results.pop(next_chunk_index):
                if cancel_event and cancel_event.is_set():
                    stop_event.set()
                    return
                yield SegmentResult(
                    index=next_segment_index,
                    start=segment.start,
                    end=segment.end,
                    text=segment.text,
                )
                next_segment_index += 1
            next_chunk_index += 1


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
    language: str = "en",
    device_ids: Optional[Sequence[int]] = None,
    cancel_event: Optional[threading.Event] = None,
):
    try:
        worker_models = _get_worker_models(model_manager, model_name, device_ids)
        primary_model = worker_models[0]
        sampling_rate = primary_model.feature_extractor.sampling_rate
        audio_input = io.BytesIO(audio_source) if isinstance(audio_source, bytes) else str(audio_source)
        audio = decode_audio(audio_input, sampling_rate=sampling_rate)
        duration = audio.shape[0] / sampling_rate
        chunks = _split_audio_with_vad(primary_model, audio)
    except Exception as exc:  # noqa: BLE001
        logger.exception(
            "Transcription setup failed model=%s device=%s device_ids=%s compute_type=%s",
            model_name,
            model_manager.device,
            list(validate_device_ids(model_manager.device, device_ids)),
            model_manager.compute_type,
        )
        raise TranscriptionError(f"Failed to decode or transcribe audio: {exc}") from exc

    yield TranscriptionInfo(duration=duration)

    if cancel_event and cancel_event.is_set():
        return

    if not chunks:
        return

    try:
        for segment in _iter_parallel_chunk_transcriptions(
            worker_models=worker_models,
            chunks=chunks,
            language=language,
            cancel_event=cancel_event,
        ):
            if cancel_event and cancel_event.is_set():
                return
            yield segment
    except Exception as exc:  # noqa: BLE001
        logger.exception(
            "Transcription execution failed model=%s device=%s device_ids=%s compute_type=%s chunks=%s",
            model_name,
            model_manager.device,
            list(validate_device_ids(model_manager.device, device_ids)),
            model_manager.compute_type,
            len(chunks),
        )
        raise TranscriptionError(f"Failed to decode or transcribe audio: {exc}") from exc
