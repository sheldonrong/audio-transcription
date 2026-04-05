from __future__ import annotations

import asyncio
import json
import logging
import mimetypes
import re
import subprocess
import threading
import uuid
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import quote

from fastapi import (
    FastAPI,
    File,
    HTTPException,
    Request,
    Response,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError
from starlette.websockets import WebSocketState

from app.schemas import (
    AcceptedEvent,
    AmdGpuInfoResponse,
    AmdGpuListResponse,
    CompleteEvent,
    ErrorEvent,
    ExportTranscriptionFileListResponse,
    ExportTranscriptionFileLoadResponse,
    ExportTranscriptionFileRequest,
    ExportTranscriptionFileResponse,
    ExportTranscriptionFileSummary,
    ProgressEvent,
    SegmentEvent,
    StartMessage,
    VideoConversionCompleteEvent,
    VideoConversionProgressEvent,
    VideoConversionStartMessage,
)
from app.transcribe import (
    ALLOWED_EXTENSIONS,
    DEFAULT_COMPUTE_TYPE,
    DEFAULT_DEVICE,
    DEFAULT_MODEL,
    MAX_UPLOAD_MB,
    MAX_UPLOAD_BYTES,
    ModelManager,
    SegmentResult,
    TranscriptionError,
    TranscriptionInfo,
    UploadValidationError,
    get_detected_amd_gpu_inventory,
    get_upload_limit_message,
    get_audio_library_dir,
    get_output_dir,
    list_audio_library_files,
    list_video_library_files,
    load_audio_from_library,
    resolve_audio_library_file,
    resolve_video_library_file,
    transcribe_audio,
    validate_upload,
    validate_upload_extension,
    validate_upload_size,
)

app = FastAPI(title="Live Transcription API", version="1.0.0")
logger = logging.getLogger(__name__)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_manager = ModelManager(device=DEFAULT_DEVICE, compute_type=DEFAULT_COMPUTE_TYPE)
detected_amd_gpu_inventory = get_detected_amd_gpu_inventory()
audio_library_dir = get_audio_library_dir()
app.mount("/audios", StaticFiles(directory=str(audio_library_dir), check_dir=False), name="audios")
output_dir = get_output_dir()
frontend_dist_dir = Path(__file__).resolve().parents[2] / "frontend" / "dist"


def slugify(value: str) -> str:
    lowered = value.strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "-", lowered).strip("-")
    return slug or "transcript"


def get_unique_transcription_file_path(directory: Path, base_name: str) -> Path:
    candidate = directory / f"{base_name}.tsp"
    if not candidate.exists():
        return candidate

    counter = 1
    while True:
        next_candidate = directory / f"{base_name}-{counter}.tsp"
        if not next_candidate.exists():
            return next_candidate
        counter += 1


def get_unique_audio_file_path(directory: Path, filename: str) -> Path:
    normalized = Path(filename)
    candidate = directory / normalized
    if not candidate.exists():
        candidate.parent.mkdir(parents=True, exist_ok=True)
        return candidate

    stem = normalized.stem or "audio"
    suffix = normalized.suffix
    parent = normalized.parent
    counter = 1
    while True:
        next_candidate = directory / parent / f"{stem}-{counter}{suffix}"
        if not next_candidate.exists():
            next_candidate.parent.mkdir(parents=True, exist_ok=True)
            return next_candidate
        counter += 1


def format_audio_output_filename(video_filename: str) -> str:
    normalized = Path(video_filename)
    stem = normalized.stem or "converted-audio"
    return (normalized.parent / f"{stem}.m4a").as_posix()


def parse_ffmpeg_out_time(value: str) -> Optional[float]:
    normalized = (value or "").strip()
    if not normalized:
        return None

    if ":" in normalized:
        parts = normalized.split(":")
        if len(parts) != 3:
            return None
        try:
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = float(parts[2])
        except ValueError:
            return None
        return (hours * 3600) + (minutes * 60) + seconds

    try:
        # ffmpeg progress output reports these fields in microseconds despite the `ms` key name.
        return float(normalized) / 1_000_000.0
    except ValueError:
        return None


def probe_media_duration(video_path: Path) -> Optional[float]:
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(video_path),
            ],
            capture_output=True,
            check=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise TranscriptionError("ffprobe is not installed or not available on PATH.") from exc
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.strip() or exc.stdout.strip() or str(exc)
        raise TranscriptionError(f"Failed to inspect uploaded video with ffprobe: {detail}") from exc

    raw_duration = result.stdout.strip()
    if not raw_duration:
        return None

    try:
        parsed = float(raw_duration)
    except ValueError:
        return None

    return parsed if parsed > 0 else None


@dataclass
class VideoConversionProgress:
    processed_seconds: float
    total_estimated_seconds: Optional[float]
    percent: Optional[float]


def convert_video_to_audio(
    video_path: Path,
    output_path: Path,
    total_duration: Optional[float],
    cancel_event: threading.Event,
):
    try:
        process = subprocess.Popen(
            [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-i",
                str(video_path),
                "-vn",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-movflags",
                "+faststart",
                "-progress",
                "pipe:1",
                "-nostats",
                str(output_path),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError as exc:
        raise TranscriptionError("ffmpeg is not installed or not available on PATH.") from exc

    last_processed_seconds = 0.0
    pending_processed_seconds: Optional[float] = None

    try:
        assert process.stdout is not None
        while True:
            if cancel_event.is_set():
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                return

            line = process.stdout.readline()
            if line == "":
                if process.poll() is not None:
                    break
                continue

            normalized = line.strip()
            if not normalized or "=" not in normalized:
                continue

            key, value = normalized.split("=", 1)
            if key in {"out_time", "out_time_us", "out_time_ms"}:
                parsed_seconds = parse_ffmpeg_out_time(value)
                if parsed_seconds is not None:
                    pending_processed_seconds = max(0.0, parsed_seconds)
                continue

            if key != "progress":
                continue

            if pending_processed_seconds is not None:
                last_processed_seconds = pending_processed_seconds

            percent: Optional[float]
            if total_duration and total_duration > 0:
                percent = max(0.0, min(100.0, (last_processed_seconds / total_duration) * 100.0))
            else:
                percent = None

            yield VideoConversionProgress(
                processed_seconds=last_processed_seconds,
                total_estimated_seconds=total_duration,
                percent=percent,
            )

            if value == "end":
                break

        stderr_output = ""
        if process.stderr is not None:
            stderr_output = process.stderr.read().strip()

        return_code = process.wait()
        if cancel_event.is_set():
            return
        if return_code != 0:
            raise TranscriptionError(stderr_output or "ffmpeg exited with a non-zero status.")

        if output_path.exists():
            final_duration = total_duration if total_duration and total_duration > 0 else last_processed_seconds or None
            yield VideoConversionProgress(
                processed_seconds=final_duration or last_processed_seconds,
                total_estimated_seconds=total_duration,
                percent=100.0 if final_duration else None,
            )
    finally:
        if process.stdout is not None:
            process.stdout.close()
        if process.stderr is not None:
            process.stderr.close()


def resolve_output_transcription_file_path(transcription_filename: str) -> Path:
    normalized = (transcription_filename or "").strip()
    if not normalized:
        raise HTTPException(status_code=400, detail="Missing Transcription File name.")
    if Path(normalized).name != normalized:
        raise HTTPException(status_code=400, detail="Invalid Transcription File name.")
    if not normalized.lower().endswith(".tsp"):
        raise HTTPException(status_code=400, detail="Expected a .tsp file.")

    target = (output_dir / normalized).resolve()
    try:
        target.relative_to(output_dir.resolve())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Transcription File path escapes OUTPUT_DIR.") from exc

    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail=f"Transcription File not found in OUTPUT_DIR: {normalized}")
    return target


def parse_transcription_file(archive_path: Path) -> dict:
    try:
        with zipfile.ZipFile(archive_path, mode="r") as archive:
            members = [name for name in archive.namelist() if name and not name.endswith("/")]
            json_member = next((name for name in members if name.lower().endswith(".json")), None)
            if not json_member:
                raise HTTPException(status_code=400, detail="Transcription File does not contain a transcript JSON file.")

            try:
                payload = json.loads(archive.read(json_member).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise HTTPException(status_code=400, detail=f"Invalid transcript JSON in Transcription File: {exc}") from exc

            if not isinstance(payload, dict):
                raise HTTPException(status_code=400, detail="Transcript JSON root must be an object.")

            raw_segments = payload.get("segments", [])
            segments: list[dict] = []
            if isinstance(raw_segments, list):
                for raw in raw_segments:
                    if not isinstance(raw, dict):
                        continue
                    try:
                        start = float(raw.get("start", 0.0))
                        end = float(raw.get("end", 0.0))
                    except (TypeError, ValueError):
                        continue
                    text = raw.get("text", "")
                    segments.append(
                        {
                            "start": start,
                            "end": end,
                            "text": text if isinstance(text, str) else "",
                        }
                    )

            transcript = payload.get("transcript")
            transcript_text = transcript.strip() if isinstance(transcript, str) else ""
            if not transcript_text:
                transcript_text = " ".join(segment["text"] for segment in segments if segment["text"]).strip()

            if not transcript_text and not segments:
                raise HTTPException(status_code=400, detail="Transcript JSON must include transcript text or segments.")

            title = payload.get("title")
            title_text = title.strip() if isinstance(title, str) else ""

            audio_filename = payload.get("audio_filename")
            audio_filename_text = audio_filename.strip() if isinstance(audio_filename, str) else ""
            audio_path = payload.get("audio_path")
            audio_path_text = audio_path.strip() if isinstance(audio_path, str) else audio_filename_text

            audio_member = ""
            for candidate in (audio_filename_text, audio_path_text):
                if candidate and candidate in members:
                    audio_member = candidate
                    break

            if not audio_member:
                preferred_names = {
                    Path(name).name
                    for name in (audio_filename_text, audio_path_text)
                    if isinstance(name, str) and name
                }
                for member in members:
                    if Path(member).name in preferred_names:
                        audio_member = member
                        break

            if not audio_member:
                for member in members:
                    if Path(member).suffix.lower() in ALLOWED_EXTENSIONS:
                        audio_member = member
                        break

            if not audio_member:
                raise HTTPException(status_code=400, detail="Transcription File does not contain a supported audio file.")

            if not audio_filename_text:
                audio_filename_text = Path(audio_member).name
            if not audio_path_text:
                audio_path_text = audio_member

            return {
                "title": title_text,
                "transcript": transcript_text,
                "segments": segments,
                "audio_filename": audio_filename_text,
                "audio_path": audio_path_text,
                "audio_member": audio_member,
            }
    except zipfile.BadZipFile as exc:
        raise HTTPException(status_code=400, detail=f"Invalid Transcription File archive: {exc}") from exc


def parse_byte_range(range_header: str, total_length: int) -> tuple[int, int]:
    content_range_any = f"bytes */{total_length}"

    def out_of_range(message: str) -> HTTPException:
        return HTTPException(status_code=416, detail=message, headers={"Content-Range": content_range_any})

    normalized = (range_header or "").strip()
    if not normalized.startswith("bytes="):
        raise out_of_range("Only byte ranges are supported.")

    spec = normalized[len("bytes=") :].strip()
    if "," in spec:
        raise out_of_range("Multiple ranges are not supported.")
    if "-" not in spec:
        raise out_of_range("Invalid byte range syntax.")

    start_raw, end_raw = spec.split("-", 1)
    start_raw = start_raw.strip()
    end_raw = end_raw.strip()

    try:
        if not start_raw:
            if not end_raw:
                raise out_of_range("Invalid suffix byte range.")
            suffix_length = int(end_raw)
            if suffix_length <= 0:
                raise out_of_range("Invalid suffix byte range.")
            if suffix_length >= total_length:
                return 0, total_length - 1
            return total_length - suffix_length, total_length - 1

        start = int(start_raw)
        if start < 0 or start >= total_length:
            raise out_of_range("Range start is out of bounds.")

        if not end_raw:
            return start, total_length - 1

        end = int(end_raw)
        if end < start:
            raise out_of_range("Range end precedes range start.")
        return start, min(end, total_length - 1)
    except ValueError as exc:
        raise out_of_range("Invalid numeric byte range.") from exc


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "defaults": {
            "model": DEFAULT_MODEL,
            "device": DEFAULT_DEVICE,
            "compute_type": DEFAULT_COMPUTE_TYPE,
            "max_upload_mb": MAX_UPLOAD_MB,
        },
    }


@app.get("/api/hardware/amd-gpus", response_model=AmdGpuListResponse)
async def list_amd_gpus() -> AmdGpuListResponse:
    return AmdGpuListResponse(
        gpus=[
            AmdGpuInfoResponse(
                device_id=gpu.device_id,
                name=gpu.name,
                gfx_version=gpu.gfx_version,
            )
            for gpu in detected_amd_gpu_inventory.gpus
        ],
        detected_at=detected_amd_gpu_inventory.detected_at,
        detection_method=detected_amd_gpu_inventory.detection_method,
        detection_error=detected_amd_gpu_inventory.detection_error,
    )


@app.get("/api/audios")
async def list_audios(request: Request) -> dict:
    files = list_audio_library_files(audio_library_dir)
    base_url = str(request.base_url).rstrip("/")

    return {
        "directory": str(audio_library_dir),
        "files": [
            {
                "filename": item.filename,
                "path": item.path,
                "size_bytes": item.size_bytes,
                "url": f"{base_url}/audios/{quote(item.path, safe='/')}",
            }
            for item in files
        ],
    }


@app.get("/api/videos")
async def list_videos(request: Request) -> dict:
    files = list_video_library_files(audio_library_dir)
    base_url = str(request.base_url).rstrip("/")

    return {
        "directory": str(audio_library_dir),
        "files": [
            {
                "filename": item.filename,
                "path": item.path,
                "size_bytes": item.size_bytes,
                "url": f"{base_url}/audios/{quote(item.path, safe='/')}",
            }
            for item in files
        ],
    }


@app.post("/api/audios/upload")
async def upload_audio(request: Request, file: UploadFile = File(...)) -> dict:
    raw_filename = (file.filename or "").strip()
    normalized_filename = Path(raw_filename.replace("\\", "/")).name
    if not normalized_filename:
        raise HTTPException(status_code=400, detail="Missing upload filename.")

    try:
        validate_upload_extension(normalized_filename)
    except UploadValidationError as exc:
        await file.close()
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    destination: Optional[Path] = None
    size_bytes = 0

    try:
        audio_library_dir.mkdir(parents=True, exist_ok=True)
        destination = get_unique_audio_file_path(audio_library_dir, normalized_filename)
        with destination.open("wb") as output_stream:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                next_size = size_bytes + len(chunk)
                if next_size > MAX_UPLOAD_BYTES:
                    raise UploadValidationError(get_upload_limit_message())
                output_stream.write(chunk)
                size_bytes = next_size
        validate_upload_size(size_bytes)
    except UploadValidationError as exc:
        if destination and destination.exists():
            destination.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except OSError as exc:
        if destination and destination.exists():
            destination.unlink(missing_ok=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to write uploaded audio file to AUDIO_DIR: {exc}",
        ) from exc
    finally:
        await file.close()

    relative_path = destination.relative_to(audio_library_dir).as_posix()
    base_url = str(request.base_url).rstrip("/")
    return {
        "filename": destination.name,
        "path": relative_path,
        "size_bytes": size_bytes,
        "url": f"{base_url}/audios/{quote(relative_path, safe='/')}",
    }


@app.get("/api/exports", response_model=ExportTranscriptionFileListResponse)
async def list_transcription_files(request: Request) -> ExportTranscriptionFileListResponse:
    base_url = str(request.base_url).rstrip("/")

    if not output_dir.exists() or not output_dir.is_dir():
        return ExportTranscriptionFileListResponse(directory=str(output_dir), files=[])

    transcription_files = [
        path
        for path in output_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".tsp"
    ]
    transcription_files.sort(key=lambda path: path.stat().st_mtime, reverse=True)

    summaries = [
        ExportTranscriptionFileSummary(
            filename=path.name,
            size_bytes=path.stat().st_size,
            modified_at=datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat(),
            url=f"{base_url}/api/exports/{quote(path.name)}",
        )
        for path in transcription_files
    ]

    return ExportTranscriptionFileListResponse(directory=str(output_dir), files=summaries)


@app.get("/api/exports/{transcription_filename}", response_model=ExportTranscriptionFileLoadResponse)
async def load_transcription_file(
    transcription_filename: str, request: Request
) -> ExportTranscriptionFileLoadResponse:
    archive_path = resolve_output_transcription_file_path(transcription_filename)
    parsed = parse_transcription_file(archive_path)
    base_url = str(request.base_url).rstrip("/")
    audio_member = parsed["audio_member"]
    audio_url = (
        f"{base_url}/api/exports/{quote(transcription_filename)}/audio"
        f"?member={quote(audio_member, safe='')}"
    )

    return ExportTranscriptionFileLoadResponse(
        transcription_file_filename=archive_path.name,
        title=parsed["title"],
        transcript=parsed["transcript"],
        segments=parsed["segments"],
        audio_filename=parsed["audio_filename"],
        audio_path=parsed["audio_path"],
        audio_url=audio_url,
    )


@app.get("/api/exports/{transcription_filename}/audio")
async def stream_transcription_file_audio(
    transcription_filename: str,
    member: str,
    request: Request,
) -> Response:
    archive_path = resolve_output_transcription_file_path(transcription_filename)
    member_name = (member or "").strip()
    if not member_name:
        raise HTTPException(status_code=400, detail="Missing audio member name.")

    try:
        with zipfile.ZipFile(archive_path, mode="r") as archive:
            if member_name not in archive.namelist():
                raise HTTPException(
                    status_code=404,
                    detail=f"Audio member not found in Transcription File: {member_name}",
                )
            payload = archive.read(member_name)
    except zipfile.BadZipFile as exc:
        raise HTTPException(status_code=400, detail=f"Invalid Transcription File archive: {exc}") from exc

    media_type = mimetypes.guess_type(member_name)[0] or "application/octet-stream"
    total_length = len(payload)
    range_header = request.headers.get("range")
    default_headers = {
        "Accept-Ranges": "bytes",
        "Content-Length": str(total_length),
    }

    if not range_header:
        return Response(content=payload, media_type=media_type, headers=default_headers)

    start, end = parse_byte_range(range_header, total_length)
    chunk = payload[start : end + 1]
    partial_headers = {
        "Accept-Ranges": "bytes",
        "Content-Range": f"bytes {start}-{end}/{total_length}",
        "Content-Length": str(len(chunk)),
    }
    return Response(content=chunk, media_type=media_type, headers=partial_headers, status_code=206)


@app.post(
    "/api/exports/transcription-file",
    response_model=ExportTranscriptionFileResponse,
)
async def export_transcription_file(
    payload: ExportTranscriptionFileRequest,
) -> ExportTranscriptionFileResponse:
    try:
        resolved_audio_filename, audio_bytes = load_audio_from_library(
            payload.audio_path, audio_library_dir
        )
    except UploadValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    transcript_text = " ".join(segment.text.strip() for segment in payload.segments if segment.text).strip()
    json_payload = {
        "title": payload.title.strip(),
        "audio_filename": resolved_audio_filename,
        # Keep path relative to the archive so the pair can move together.
        "audio_path": resolved_audio_filename,
        "transcript": transcript_text,
        "segments": [
            {"start": segment.start, "end": segment.end, "text": segment.text}
            for segment in payload.segments
        ],
    }

    safe_title = slugify(payload.title)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    archive_base = f"{safe_title}-{timestamp}"
    json_filename = f"{safe_title}-transcript.json"

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        archive_path = get_unique_transcription_file_path(output_dir, archive_base)
        with zipfile.ZipFile(archive_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(json_filename, json.dumps(json_payload, indent=2))
            archive.writestr(resolved_audio_filename, audio_bytes)
    except OSError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to write Transcription File to OUTPUT_DIR: {exc}",
        ) from exc

    return ExportTranscriptionFileResponse(
        transcription_file_filename=archive_path.name,
        transcription_file_path=str(archive_path),
        json_filename=json_filename,
        audio_filename=resolved_audio_filename,
    )


async def send_error(websocket: WebSocket, job_id: str, message: str) -> None:
    await websocket.send_json(ErrorEvent(job_id=job_id, message=message).model_dump())


async def send_error_and_close(
    websocket: WebSocket,
    job_id: str,
    message: str,
    close_code: int,
) -> None:
    if websocket.client_state != WebSocketState.CONNECTED:
        return
    try:
        await send_error(websocket, job_id, message)
        await websocket.close(code=close_code)
    except RuntimeError:
        # Connection already closed by the client.
        return


async def receive_start_message(websocket: WebSocket) -> StartMessage:
    raw = await websocket.receive_text()
    payload = json.loads(raw)
    return StartMessage.model_validate(payload)


async def receive_video_conversion_start_message(websocket: WebSocket) -> VideoConversionStartMessage:
    raw = await websocket.receive_text()
    payload = json.loads(raw)
    return VideoConversionStartMessage.model_validate(payload)


async def receive_audio_payload(websocket: WebSocket) -> bytes:
    buffer = bytearray()

    while True:
        message = await websocket.receive()
        message_type = message.get("type")

        if message_type == "websocket.disconnect":
            raise WebSocketDisconnect(code=message.get("code", 1006))

        if message.get("bytes") is not None:
            chunk = message["bytes"]
            buffer.extend(chunk)
            if len(buffer) > MAX_UPLOAD_BYTES:
                raise UploadValidationError(get_upload_limit_message())
            continue

        text = message.get("text")
        if text is None:
            continue

        # Client signals completion of binary upload.
        if text == "__end__":
            break

    return bytes(buffer)


async def receive_binary_upload_to_path(websocket: WebSocket, destination: Path) -> int:
    size_bytes = 0

    with destination.open("wb") as output_stream:
        while True:
            message = await websocket.receive()
            message_type = message.get("type")

            if message_type == "websocket.disconnect":
                raise WebSocketDisconnect(code=message.get("code", 1006))

            if message.get("bytes") is not None:
                chunk = message["bytes"]
                next_size = size_bytes + len(chunk)
                if next_size > MAX_UPLOAD_BYTES:
                    raise UploadValidationError(get_upload_limit_message())
                output_stream.write(chunk)
                size_bytes = next_size
                continue

            text = message.get("text")
            if text is None:
                continue

            if text == "__end__":
                break

    validate_upload_size(size_bytes)
    return size_bytes


async def watch_for_disconnect(websocket: WebSocket, cancel_event: threading.Event) -> None:
    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                cancel_event.set()
                return
    except RuntimeError:
        cancel_event.set()


async def iter_transcription_events(
    audio_source: bytes | str,
    model_name: str,
    language: Optional[str],
    device_ids: list[int],
    cancel_event: threading.Event,
):
    queue: asyncio.Queue[object] = asyncio.Queue()
    sentinel = object()
    loop = asyncio.get_running_loop()

    def worker() -> None:
        try:
            for item in transcribe_audio(
                model_manager=model_manager,
                audio_source=audio_source,
                model_name=model_name,
                language=language,
                device_ids=device_ids,
                cancel_event=cancel_event,
            ):
                if cancel_event.is_set():
                    break
                loop.call_soon_threadsafe(queue.put_nowait, item)
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "Transcription worker failed model=%s device_ids=%s",
                model_name,
                device_ids,
            )
            loop.call_soon_threadsafe(queue.put_nowait, exc)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, sentinel)

    threading.Thread(target=worker, daemon=True).start()

    while True:
        if cancel_event.is_set() and queue.empty():
            break
        try:
            item = await asyncio.wait_for(queue.get(), timeout=0.2)
        except asyncio.TimeoutError:
            continue
        if item is sentinel:
            break
        if isinstance(item, Exception):
            raise item
        yield item


async def iter_video_conversion_events(
    video_path: Path,
    output_path: Path,
    total_duration: Optional[float],
    cancel_event: threading.Event,
):
    queue: asyncio.Queue[object] = asyncio.Queue()
    sentinel = object()
    loop = asyncio.get_running_loop()

    def worker() -> None:
        try:
            for item in convert_video_to_audio(
                video_path=video_path,
                output_path=output_path,
                total_duration=total_duration,
                cancel_event=cancel_event,
            ):
                if cancel_event.is_set():
                    break
                loop.call_soon_threadsafe(queue.put_nowait, item)
        except Exception as exc:  # noqa: BLE001
            loop.call_soon_threadsafe(queue.put_nowait, exc)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, sentinel)

    threading.Thread(target=worker, daemon=True).start()

    while True:
        if cancel_event.is_set() and queue.empty():
            break
        try:
            item = await asyncio.wait_for(queue.get(), timeout=0.2)
        except asyncio.TimeoutError:
            continue
        if item is sentinel:
            break
        if isinstance(item, Exception):
            raise item
        yield item


@app.websocket("/ws/transcribe")
async def ws_transcribe(websocket: WebSocket) -> None:
    await websocket.accept()

    job_id = str(uuid.uuid4())
    cancel_event = threading.Event()
    disconnect_task: Optional[asyncio.Task[None]] = None

    try:
        try:
            start = await receive_start_message(websocket)
        except (ValidationError, json.JSONDecodeError) as exc:
            await send_error_and_close(websocket, job_id, f"Invalid start message: {exc}", 1003)
            return

        await websocket.send_json(
            AcceptedEvent(job_id=job_id, filename=start.filename).model_dump()
        )

        if start.audio_path:
            resolved_path = resolve_audio_library_file(start.audio_path, audio_library_dir)
            resolved_filename = resolved_path.name
            audio_source: bytes | str = str(resolved_path)
        else:
            audio_bytes = await receive_audio_payload(websocket)
            validate_upload(start.filename, audio_bytes)
            audio_source = audio_bytes

        disconnect_task = asyncio.create_task(watch_for_disconnect(websocket, cancel_event))
        accumulated_text_parts: list[str] = []
        total_duration: Optional[float] = None
        segment_count = 0

        async for item in iter_transcription_events(
            audio_source=audio_source,
            model_name=start.model or DEFAULT_MODEL,
            language=start.language,
            device_ids=start.device_ids,
            cancel_event=cancel_event,
        ):
            if isinstance(item, TranscriptionInfo):
                total_duration = item.duration
                await websocket.send_json(
                    ProgressEvent(
                        job_id=job_id,
                        percent=0.0 if total_duration else None,
                        processed_seconds=0.0,
                        total_estimated_seconds=total_duration,
                    ).model_dump()
                )
                continue

            if isinstance(item, SegmentResult):
                segment_count += 1
                if item.text:
                    accumulated_text_parts.append(item.text)
                accumulated_text = " ".join(accumulated_text_parts).strip()

                percent: Optional[float]
                if total_duration and total_duration > 0:
                    percent = max(0.0, min(100.0, (item.end / total_duration) * 100.0))
                else:
                    percent = None

                await websocket.send_json(
                    ProgressEvent(
                        job_id=job_id,
                        percent=percent,
                        processed_seconds=item.end,
                        total_estimated_seconds=total_duration,
                    ).model_dump()
                )
                await websocket.send_json(
                    SegmentEvent(
                        job_id=job_id,
                        index=item.index,
                        start=item.start,
                        end=item.end,
                        text=item.text,
                        accumulated_text=accumulated_text,
                    ).model_dump()
                )
                await asyncio.sleep(0)

        if cancel_event.is_set():
            return

        await websocket.send_json(
            CompleteEvent(
                job_id=job_id,
                text=" ".join(accumulated_text_parts).strip(),
                segments_count=segment_count,
                duration_seconds=total_duration,
            ).model_dump()
        )
        await websocket.close(code=1000)

    except UploadValidationError as exc:
        await send_error_and_close(websocket, job_id, str(exc), 1009)
    except TranscriptionError as exc:
        await send_error_and_close(websocket, job_id, str(exc), 1011)
    except WebSocketDisconnect:
        cancel_event.set()
        return
    except Exception as exc:  # noqa: BLE001
        await send_error_and_close(websocket, job_id, f"Unexpected server error: {exc}", 1011)
    finally:
        if disconnect_task:
            disconnect_task.cancel()
            try:
                await disconnect_task
            except asyncio.CancelledError:
                pass


@app.websocket("/ws/v2a")
async def ws_video_to_audio(websocket: WebSocket) -> None:
    await websocket.accept()

    job_id = str(uuid.uuid4())
    cancel_event = threading.Event()
    disconnect_task: Optional[asyncio.Task[None]] = None
    resolved_video_path: Optional[Path] = None
    output_path: Optional[Path] = None
    completed = False

    try:
        try:
            start = await receive_video_conversion_start_message(websocket)
        except (ValidationError, json.JSONDecodeError) as exc:
            await send_error_and_close(websocket, job_id, f"Invalid start message: {exc}", 1003)
            return

        try:
            resolved_video_path = resolve_video_library_file(start.video_path, audio_library_dir)
        except UploadValidationError as exc:
            await send_error_and_close(websocket, job_id, str(exc), 1003)
            return

        await websocket.send_json(
            AcceptedEvent(job_id=job_id, filename=resolved_video_path.name).model_dump()
        )

        audio_library_dir.mkdir(parents=True, exist_ok=True)
        total_duration = probe_media_duration(resolved_video_path)
        output_path = get_unique_audio_file_path(
            audio_library_dir,
            format_audio_output_filename(start.video_path),
        )
        disconnect_task = asyncio.create_task(watch_for_disconnect(websocket, cancel_event))
        await websocket.send_json(
            VideoConversionProgressEvent(
                job_id=job_id,
                percent=0.0 if total_duration else None,
                processed_seconds=0.0,
                total_estimated_seconds=total_duration,
            ).model_dump()
        )

        async for item in iter_video_conversion_events(
            video_path=resolved_video_path,
            output_path=output_path,
            total_duration=total_duration,
            cancel_event=cancel_event,
        ):
            await websocket.send_json(
                VideoConversionProgressEvent(
                    job_id=job_id,
                    percent=item.percent,
                    processed_seconds=item.processed_seconds,
                    total_estimated_seconds=item.total_estimated_seconds,
                ).model_dump()
            )
            await asyncio.sleep(0)

        if cancel_event.is_set():
            return

        base_url = str(websocket.base_url).rstrip("/")
        relative_path = output_path.relative_to(audio_library_dir).as_posix()
        await websocket.send_json(
            VideoConversionCompleteEvent(
                job_id=job_id,
                filename=output_path.name,
                path=relative_path,
                url=f"{base_url}/audios/{quote(relative_path, safe='/')}",
                duration_seconds=total_duration,
            ).model_dump()
        )
        completed = True
        await websocket.close(code=1000)

    except UploadValidationError as exc:
        await send_error_and_close(websocket, job_id, str(exc), 1009)
    except TranscriptionError as exc:
        await send_error_and_close(websocket, job_id, str(exc), 1011)
    except WebSocketDisconnect:
        cancel_event.set()
        return
    except Exception as exc:  # noqa: BLE001
        await send_error_and_close(websocket, job_id, f"Unexpected server error: {exc}", 1011)
    finally:
        cancel_event.set()
        if disconnect_task:
            disconnect_task.cancel()
            try:
                await disconnect_task
            except asyncio.CancelledError:
                pass
        if output_path and output_path.exists() and not completed:
            output_path.unlink(missing_ok=True)


if frontend_dist_dir.exists():
    # Serve the compiled frontend from the same container/process when available.
    app.mount("/", StaticFiles(directory=str(frontend_dist_dir), html=True), name="frontend")
