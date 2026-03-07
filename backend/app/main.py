from __future__ import annotations

import asyncio
import json
import mimetypes
import re
import threading
import uuid
import zipfile
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
)
from app.transcribe import (
    ALLOWED_EXTENSIONS,
    DEFAULT_COMPUTE_TYPE,
    DEFAULT_DEVICE,
    DEFAULT_MODEL,
    MAX_UPLOAD_BYTES,
    ModelManager,
    SegmentResult,
    TranscriptionError,
    TranscriptionInfo,
    UploadValidationError,
    get_audio_library_dir,
    get_output_dir,
    list_audio_library_files,
    load_audio_from_library,
    transcribe_audio,
    validate_upload,
)

app = FastAPI(title="Live Transcription API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_manager = ModelManager(device=DEFAULT_DEVICE, compute_type=DEFAULT_COMPUTE_TYPE)
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
    candidate = directory / filename
    if not candidate.exists():
        return candidate

    stem = Path(filename).stem or "audio"
    suffix = Path(filename).suffix
    counter = 1
    while True:
        next_candidate = directory / f"{stem}-{counter}{suffix}"
        if not next_candidate.exists():
            return next_candidate
        counter += 1


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
            "max_upload_mb": MAX_UPLOAD_BYTES // (1024 * 1024),
        },
    }


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


@app.post("/api/audios/upload")
async def upload_audio(request: Request, file: UploadFile = File(...)) -> dict:
    raw_filename = (file.filename or "").strip()
    normalized_filename = Path(raw_filename.replace("\\", "/")).name
    if not normalized_filename:
        raise HTTPException(status_code=400, detail="Missing upload filename.")

    try:
        payload = await file.read()
    finally:
        await file.close()

    try:
        validate_upload(normalized_filename, payload)
    except UploadValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        audio_library_dir.mkdir(parents=True, exist_ok=True)
        destination = get_unique_audio_file_path(audio_library_dir, normalized_filename)
        destination.write_bytes(payload)
    except OSError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to write uploaded audio file to AUDIO_DIR: {exc}",
        ) from exc

    relative_path = destination.relative_to(audio_library_dir).as_posix()
    base_url = str(request.base_url).rstrip("/")
    return {
        "filename": destination.name,
        "path": relative_path,
        "size_bytes": len(payload),
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
                raise UploadValidationError("Uploaded file exceeds the 50 MB limit.")
            continue

        text = message.get("text")
        if text is None:
            continue

        # Client signals completion of binary upload.
        if text == "__end__":
            break

    return bytes(buffer)


async def iter_transcription_events(
    audio_bytes: bytes,
    model_name: str,
    language: Optional[str],
):
    queue: asyncio.Queue[object] = asyncio.Queue()
    sentinel = object()
    loop = asyncio.get_running_loop()

    def worker() -> None:
        try:
            for item in transcribe_audio(
                model_manager=model_manager,
                audio_bytes=audio_bytes,
                model_name=model_name,
                language=language,
            ):
                loop.call_soon_threadsafe(queue.put_nowait, item)
        except Exception as exc:  # noqa: BLE001
            loop.call_soon_threadsafe(queue.put_nowait, exc)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, sentinel)

    threading.Thread(target=worker, daemon=True).start()

    while True:
        item = await queue.get()
        if item is sentinel:
            break
        if isinstance(item, Exception):
            raise item
        yield item


@app.websocket("/ws/transcribe")
async def ws_transcribe(websocket: WebSocket) -> None:
    await websocket.accept()

    job_id = str(uuid.uuid4())

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
            resolved_filename, audio_bytes = load_audio_from_library(
                start.audio_path, audio_library_dir
            )
            validate_upload(resolved_filename, audio_bytes)
        else:
            audio_bytes = await receive_audio_payload(websocket)
            validate_upload(start.filename, audio_bytes)

        accumulated_text_parts: list[str] = []
        total_duration: Optional[float] = None
        segment_count = 0

        async for item in iter_transcription_events(
            audio_bytes=audio_bytes,
            model_name=start.model or DEFAULT_MODEL,
            language=start.language,
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
        return
    except Exception as exc:  # noqa: BLE001
        await send_error_and_close(websocket, job_id, f"Unexpected server error: {exc}", 1011)


if frontend_dist_dir.exists():
    # Serve the compiled frontend from the same container/process when available.
    app.mount("/", StaticFiles(directory=str(frontend_dist_dir), html=True), name="frontend")
