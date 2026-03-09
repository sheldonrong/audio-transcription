from __future__ import annotations

import json
import os
import threading
import zipfile
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app
from app.transcribe import (
    AudioLibraryFile,
    MAX_UPLOAD_MB,
    MAX_UPLOAD_BYTES,
    ModelManager,
    SegmentResult,
    TranscriptionInfo,
    apply_rocm_env_overrides,
    get_default_compute_type,
    get_default_device,
    get_runtime_device,
    is_cpu_device,
    normalize_device,
    transcribe_audio,
    validate_upload,
)


def test_validate_upload_size_limit() -> None:
    too_large = b"a" * (MAX_UPLOAD_BYTES + 1)
    try:
        validate_upload("sample.wav", too_large)
        assert False, "Expected size validation error"
    except Exception as exc:  # noqa: BLE001
        assert f"{MAX_UPLOAD_MB} MB" in str(exc)


def test_validate_upload_extension() -> None:
    try:
        validate_upload("sample.txt", b"abc")
        assert False, "Expected extension validation error"
    except Exception as exc:  # noqa: BLE001
        assert "Unsupported file extension" in str(exc)


def test_validate_upload_mobile_memo_extensions() -> None:
    validate_upload("voice-note.m4a", b"abc")
    validate_upload("voice-note.3gp", b"abc")
    validate_upload("voice-note.amr", b"abc")
    validate_upload("voice-note.opus", b"abc")


def test_transcribe_audio_respects_cancel_event() -> None:
    class FakeSegment:
        start = 0.0
        end = 1.0
        text = "hello"

    class FakeInfo:
        duration = 12.0

    class FakeModel:
        def transcribe(self, *_args, **_kwargs):
            return [FakeSegment()], FakeInfo()

    class FakeManager:
        def get(self, _model_name: str):
            return FakeModel()

    cancel_event = threading.Event()
    events = transcribe_audio(
        model_manager=FakeManager(),  # type: ignore[arg-type]
        audio_source=b"fake audio",
        model_name="medium",
        language="en",
        cancel_event=cancel_event,
    )

    first_event = next(events)
    assert isinstance(first_event, TranscriptionInfo)

    cancel_event.set()
    assert list(events) == []


def test_health_endpoint() -> None:
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["defaults"]["model"] == "medium"
    assert data["defaults"]["max_upload_mb"] == MAX_UPLOAD_MB


def test_cpu_compute_type_defaults_to_int8() -> None:
    assert is_cpu_device("cpu")
    assert is_cpu_device("CPU")
    assert get_default_compute_type("cpu") == "int8"


def test_non_cpu_compute_type_defaults_to_float16() -> None:
    assert not is_cpu_device("cuda")
    assert get_default_compute_type("cuda") == "float16"
    assert get_default_compute_type("rocm") == "float16"


def test_compute_type_env_override(monkeypatch) -> None:
    monkeypatch.setenv("WHISPER_COMPUTE_TYPE", "float32")
    assert get_default_compute_type("cpu") == "float32"


def test_device_env_override(monkeypatch) -> None:
    monkeypatch.setenv("WHISPER_DEVICE", "cpu")
    assert get_default_device() == "cpu"


def test_model_manager_uses_int8_when_device_is_cpu() -> None:
    manager = ModelManager(device="cpu")
    assert manager.device == "cpu"
    assert manager.compute_type == "int8"


def test_rocm_device_maps_to_cuda_runtime() -> None:
    manager = ModelManager(device="rocm")
    assert manager.device == "rocm"
    assert manager.runtime_device == "cuda"
    assert manager.compute_type == "float16"


def test_device_normalization() -> None:
    assert normalize_device(" CPU ") == "cpu"
    assert get_runtime_device("rocm") == "cuda"


def test_apply_rocm_env_overrides(monkeypatch) -> None:
    monkeypatch.delenv("HSA_OVERRIDE_GFX_VERSION", raising=False)
    monkeypatch.delenv("ROCR_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv("WHISPER_HSA_OVERRIDE_GFX_VERSION", "9.0.0")
    monkeypatch.setenv("WHISPER_ROCR_VISIBLE_DEVICES", "0")

    apply_rocm_env_overrides()

    assert os.getenv("HSA_OVERRIDE_GFX_VERSION") == "9.0.0"
    assert os.getenv("ROCR_VISIBLE_DEVICES") == "0"


def test_list_audios_endpoint(monkeypatch) -> None:
    monkeypatch.setattr("app.main.audio_library_dir", Path("/audios"))
    monkeypatch.setattr(
        "app.main.list_audio_library_files",
        lambda _: [
            AudioLibraryFile(filename="sample.wav", path="sample.wav", size_bytes=1234),
            AudioLibraryFile(filename="nested.mp3", path="set/nested.mp3", size_bytes=4096),
        ],
    )

    client = TestClient(app)
    response = client.get("/api/audios")
    assert response.status_code == 200
    body = response.json()
    assert body["directory"] == "/audios"
    assert len(body["files"]) == 2
    assert body["files"][0]["url"].endswith("/audios/sample.wav")
    assert body["files"][1]["url"].endswith("/audios/set/nested.mp3")


def test_upload_audio_endpoint(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("app.main.audio_library_dir", tmp_path)

    client = TestClient(app)
    response = client.post(
        "/api/audios/upload",
        files={"file": ("meeting.wav", b"fake audio payload", "audio/wav")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["filename"] == "meeting.wav"
    assert body["path"] == "meeting.wav"
    assert body["size_bytes"] == len(b"fake audio payload")
    assert body["url"].endswith("/audios/meeting.wav")
    assert (tmp_path / "meeting.wav").read_bytes() == b"fake audio payload"


def test_upload_audio_endpoint_with_duplicate_name(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("app.main.audio_library_dir", tmp_path)
    (tmp_path / "meeting.wav").write_bytes(b"existing")

    client = TestClient(app)
    response = client.post(
        "/api/audios/upload",
        files={"file": ("meeting.wav", b"new audio payload", "audio/wav")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["filename"] == "meeting-1.wav"
    assert body["path"] == "meeting-1.wav"
    assert (tmp_path / "meeting.wav").read_bytes() == b"existing"
    assert (tmp_path / "meeting-1.wav").read_bytes() == b"new audio payload"


def test_upload_audio_endpoint_rejects_non_audio(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("app.main.audio_library_dir", tmp_path)

    client = TestClient(app)
    response = client.post(
        "/api/audios/upload",
        files={"file": ("notes.txt", b"not audio", "text/plain")},
    )

    assert response.status_code == 400
    body = response.json()
    assert "Unsupported file extension" in body["detail"]
    assert not (tmp_path / "notes.txt").exists()


def test_ws_transcribe_success(monkeypatch) -> None:
    def fake_transcribe_audio(*args, **kwargs):
        yield TranscriptionInfo(duration=10.0)
        yield SegmentResult(index=0, start=0.0, end=3.0, text="hello")
        yield SegmentResult(index=1, start=3.0, end=10.0, text="world")

    monkeypatch.setattr("app.main.transcribe_audio", fake_transcribe_audio)

    client = TestClient(app)
    with client.websocket_connect("/ws/transcribe") as ws:
        ws.send_json(
            {
                "type": "start",
                "filename": "sample.wav",
                "language": "en",
                "model": "medium",
            }
        )
        ws.send_bytes(b"fake audio")
        ws.send_text("__end__")

        accepted = ws.receive_json()
        progress0 = ws.receive_json()
        progress1 = ws.receive_json()
        segment1 = ws.receive_json()
        progress2 = ws.receive_json()
        segment2 = ws.receive_json()
        complete = ws.receive_json()

        assert accepted["type"] == "accepted"
        assert progress0["type"] == "progress"
        assert progress1["percent"] == 30.0
        assert segment2["accumulated_text"] == "hello world"
        assert complete["type"] == "complete"
        assert complete["segments_count"] == 2


def test_ws_transcribe_from_audio_library(monkeypatch) -> None:
    called: dict[str, str | None] = {"path": None}

    def fake_load_audio_from_library(path: str, *_args):
        called["path"] = path
        return "sample.wav", b"fake audio"

    def fake_transcribe_audio(*args, **kwargs):
        yield TranscriptionInfo(duration=5.0)
        yield SegmentResult(index=0, start=0.0, end=5.0, text="from library")

    monkeypatch.setattr("app.main.load_audio_from_library", fake_load_audio_from_library)
    monkeypatch.setattr("app.main.transcribe_audio", fake_transcribe_audio)

    client = TestClient(app)
    with client.websocket_connect("/ws/transcribe") as ws:
        ws.send_json(
            {
                "type": "start",
                "filename": "sample.wav",
                "audio_path": "sample.wav",
                "language": "en",
                "model": "medium",
            }
        )

        accepted = ws.receive_json()
        progress0 = ws.receive_json()
        progress1 = ws.receive_json()
        segment1 = ws.receive_json()
        complete = ws.receive_json()

        assert called["path"] == "sample.wav"
        assert accepted["type"] == "accepted"
        assert progress0["type"] == "progress"
        assert progress1["percent"] == 100.0
        assert segment1["accumulated_text"] == "from library"
        assert complete["type"] == "complete"
        assert complete["segments_count"] == 1


def test_export_transcription_file(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("app.main.output_dir", tmp_path)

    def fake_load_audio_from_library(path: str, *_args):
        assert path == "sample.wav"
        return "sample.wav", b"fake audio data"

    monkeypatch.setattr("app.main.load_audio_from_library", fake_load_audio_from_library)

    client = TestClient(app)
    response = client.post(
        "/api/exports/transcription-file",
        json={
            "title": "My Test Transcript",
            "audio_filename": "sample.wav",
            "audio_path": "sample.wav",
            "segments": [
                {"start": 0.0, "end": 1.0, "text": "hello"},
                {"start": 1.0, "end": 2.0, "text": "world"},
            ],
        },
    )

    assert response.status_code == 200
    body = response.json()
    transcription_file_path = Path(body["transcription_file_path"])
    assert transcription_file_path.exists()
    assert transcription_file_path.parent == tmp_path
    assert transcription_file_path.suffix == ".tsp"

    with zipfile.ZipFile(transcription_file_path, "r") as archive:
        names = set(archive.namelist())
        assert body["json_filename"] in names
        assert body["audio_filename"] in names

        parsed_json = json.loads(archive.read(body["json_filename"]).decode("utf-8"))
        assert parsed_json["title"] == "My Test Transcript"
        assert parsed_json["audio_filename"] == "sample.wav"
        assert parsed_json["audio_path"] == "sample.wav"
        assert parsed_json["transcript"] == "hello world"
        assert len(parsed_json["segments"]) == 2


def test_list_transcription_files(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("app.main.output_dir", tmp_path)

    first = tmp_path / "first.tsp"
    second = tmp_path / "second.tsp"
    first.write_bytes(b"pk")
    second.write_bytes(b"pk")

    client = TestClient(app)
    response = client.get("/api/exports")
    assert response.status_code == 200
    body = response.json()
    assert body["directory"] == str(tmp_path)
    listed = {item["filename"] for item in body["files"]}
    assert listed == {"first.tsp", "second.tsp"}


def test_load_transcription_file_and_audio_stream(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("app.main.output_dir", tmp_path)

    archive_path = tmp_path / "sample-export.tsp"
    payload = {
        "title": "Loaded Title",
        "audio_filename": "sample.wav",
        "audio_path": "sample.wav",
        "segments": [
            {"start": 0.0, "end": 1.5, "text": "hello"},
            {"start": 1.5, "end": 3.0, "text": "world"},
        ],
    }
    with zipfile.ZipFile(archive_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("sample-transcript.json", json.dumps(payload))
        archive.writestr("sample.wav", b"audio bytes")

    client = TestClient(app)
    response = client.get("/api/exports/sample-export.tsp")
    assert response.status_code == 200
    body = response.json()
    assert body["transcription_file_filename"] == "sample-export.tsp"
    assert body["title"] == "Loaded Title"
    assert len(body["segments"]) == 2
    assert body["audio_filename"] == "sample.wav"
    assert "audio?member=" in body["audio_url"]

    audio_response = client.get(
        "/api/exports/sample-export.tsp/audio",
        params={"member": "sample.wav"},
    )
    assert audio_response.status_code == 200
    assert audio_response.content == b"audio bytes"
    assert audio_response.headers.get("accept-ranges") == "bytes"

    partial_response = client.get(
        "/api/exports/sample-export.tsp/audio",
        params={"member": "sample.wav"},
        headers={"Range": "bytes=6-9"},
    )
    assert partial_response.status_code == 206
    assert partial_response.content == b"byte"
    assert partial_response.headers.get("content-range") == "bytes 6-9/11"

    suffix_response = client.get(
        "/api/exports/sample-export.tsp/audio",
        params={"member": "sample.wav"},
        headers={"Range": "bytes=-5"},
    )
    assert suffix_response.status_code == 206
    assert suffix_response.content == b"bytes"
    assert suffix_response.headers.get("content-range") == "bytes 6-10/11"

    invalid_range_response = client.get(
        "/api/exports/sample-export.tsp/audio",
        params={"member": "sample.wav"},
        headers={"Range": "bytes=20-30"},
    )
    assert invalid_range_response.status_code == 416


def test_ws_transcribe_invalid_start_message() -> None:
    client = TestClient(app)
    with client.websocket_connect("/ws/transcribe") as ws:
        ws.send_text("not json")
        error = ws.receive_json()
        assert error["type"] == "error"
        assert "Invalid start message" in error["message"]
