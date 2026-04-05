from __future__ import annotations

import json
import threading
import zipfile
from pathlib import Path

import numpy as np
import app.transcribe as transcribe_module
from fastapi.testclient import TestClient

from app.main import VideoConversionProgress, app
from app.transcribe import (
    AmdGpuInfo,
    AmdGpuInventory,
    AudioLibraryFile,
    MAX_UPLOAD_MB,
    MAX_UPLOAD_BYTES,
    ModelManager,
    SegmentResult,
    TranscriptionInfo,
    get_default_batch_size,
    get_default_compute_type,
    get_default_device,
    get_runtime_device,
    is_cpu_device,
    normalize_device,
    normalize_device_ids,
    validate_device_ids,
    transcribe_audio,
    _parse_rocm_smi_gpus,
    validate_upload,
    validate_video_upload_extension,
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


def test_validate_video_upload_extension() -> None:
    validate_video_upload_extension("clip.mp4")
    validate_video_upload_extension("clip.mkv")

    try:
        validate_video_upload_extension("clip.mov")
        assert False, "Expected video extension validation error"
    except Exception as exc:  # noqa: BLE001
        assert "Unsupported video extension" in str(exc)


def test_transcribe_audio_respects_cancel_event(monkeypatch) -> None:
    class FakeSegment:
        start = 0.0
        end = 1.0
        text = "hello"

    class FakeModel:
        feature_extractor = type("FeatureExtractor", (), {"sampling_rate": 16000})()

        def transcribe(self, *_args, **_kwargs):
            return [FakeSegment()], object()

    class FakeManager:
        device = "cuda"

        def __init__(self) -> None:
            self.model = FakeModel()

        def get(self, _model_name: str, device_ids=None):
            return self.model

    monkeypatch.setattr(
        transcribe_module,
        "decode_audio",
        lambda *_args, **_kwargs: np.zeros(16000, dtype=np.float32),
    )
    monkeypatch.setattr(
        transcribe_module,
        "_split_audio_with_vad",
        lambda *_args, **_kwargs: [
            transcribe_module.AudioChunk(
                index=0,
                waveform=np.zeros(16000, dtype=np.float32),
                speech_segments=[{"start": 0, "end": 16000}],
            )
        ],
    )
    monkeypatch.setattr(transcribe_module, "_detect_transcription_language", lambda *_args, **_kwargs: "en")
    monkeypatch.setattr(transcribe_module, "restore_speech_timestamps", lambda segments, *_args: segments)

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


def test_batch_size_defaults_to_2() -> None:
    assert get_default_batch_size() == 2


def test_batch_size_env_override(monkeypatch) -> None:
    monkeypatch.setenv("WHISPER_BATCH_SIZE", "6")
    assert get_default_batch_size() == 6


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


def test_model_manager_keeps_device_specific_models_available_for_parallel_workers(monkeypatch) -> None:
    created: list[dict[str, object]] = []

    class FakeWhisperModel:
        def __init__(self, _model_name: str, **kwargs):
            created.append(kwargs)

    monkeypatch.setattr(transcribe_module, "WhisperModel", FakeWhisperModel)
    monkeypatch.setattr(
        transcribe_module,
        "get_detected_amd_gpu_inventory",
        lambda: AmdGpuInventory(
            gpus=[
                AmdGpuInfo(device_id=0, name="GPU 0"),
                AmdGpuInfo(device_id=1, name="GPU 1"),
            ],
            detected_at="2026-04-03T00:00:00+00:00",
        ),
    )

    manager = ModelManager(device="rocm")
    manager.get("medium", device_ids=[0])
    manager.get("medium", device_ids=[1])
    manager.get("medium", device_ids=[0, 1])

    assert len(manager._models) == 3
    assert ("medium", (0,)) in manager._models
    assert ("medium", (1,)) in manager._models
    assert ("medium", (0, 1)) in manager._models
    assert created[0]["device_index"] == 0
    assert created[1]["device_index"] == 1
    assert created[2]["device_index"] == [0, 1]
    assert created[2]["num_workers"] == 2


def test_transcribe_audio_uses_whisper_model_api_with_manual_vad_chunks(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeSegment:
        start = 0.0
        end = 1.0
        text = "hello"

    class FakeModel:
        feature_extractor = type("FeatureExtractor", (), {"sampling_rate": 16000})()

        def transcribe(self, *_args, **kwargs):
            captured.update(kwargs)
            return [FakeSegment()], object()

    class FakeManager:
        device = "cuda"

        def __init__(self) -> None:
            self.model = FakeModel()

        def get(self, _model_name: str, device_ids=None):
            return self.model

    monkeypatch.setattr(
        transcribe_module,
        "decode_audio",
        lambda *_args, **_kwargs: np.zeros(16000, dtype=np.float32),
    )
    monkeypatch.setattr(
        transcribe_module,
        "_split_audio_with_vad",
        lambda *_args, **_kwargs: [
            transcribe_module.AudioChunk(
                index=0,
                waveform=np.zeros(16000, dtype=np.float32),
                speech_segments=[{"start": 0, "end": 16000}],
            )
        ],
    )
    monkeypatch.setattr(transcribe_module, "_detect_transcription_language", lambda *_args, **_kwargs: "en")
    monkeypatch.setattr(transcribe_module, "restore_speech_timestamps", lambda segments, *_args: segments)

    events = list(
        transcribe_audio(
            model_manager=FakeManager(),  # type: ignore[arg-type]
            audio_source=b"fake audio",
            model_name="medium",
            language="en",
        )
    )

    assert len(events) == 2
    assert captured["language"] == "en"
    assert captured["vad_filter"] is False
    assert captured["condition_on_previous_text"] is False


def test_parallel_chunk_scheduler_dispatches_next_chunk_when_worker_finishes(monkeypatch) -> None:
    release_first_chunk = threading.Event()
    third_chunk_started = threading.Event()
    results: list[SegmentResult] = []
    errors: list[Exception] = []

    class FakeSegment:
        def __init__(self, chunk_id: int) -> None:
            self.start = float(chunk_id)
            self.end = float(chunk_id) + 0.5
            self.text = f"chunk-{chunk_id}"

    class FakeModel:
        feature_extractor = type("FeatureExtractor", (), {"sampling_rate": 16000})()

        def transcribe(self, waveform, **_kwargs):
            chunk_id = int(waveform[0])
            if chunk_id == 0:
                assert release_first_chunk.wait(timeout=2.0)
            if chunk_id == 2:
                third_chunk_started.set()
            return [FakeSegment(chunk_id)], object()

    monkeypatch.setattr(transcribe_module, "restore_speech_timestamps", lambda segments, *_args: segments)

    chunks = [
        transcribe_module.AudioChunk(index=0, waveform=np.array([0], dtype=np.float32), speech_segments=[{"start": 0, "end": 1}]),
        transcribe_module.AudioChunk(index=1, waveform=np.array([1], dtype=np.float32), speech_segments=[{"start": 1, "end": 2}]),
        transcribe_module.AudioChunk(index=2, waveform=np.array([2], dtype=np.float32), speech_segments=[{"start": 2, "end": 3}]),
    ]

    def consume() -> None:
        try:
            results.extend(
                list(
                    transcribe_module._iter_parallel_chunk_transcriptions(
                        worker_models=[FakeModel(), FakeModel()],
                        chunks=chunks,
                        language="en",
                        cancel_event=None,
                    )
                )
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    consumer = threading.Thread(target=consume)
    consumer.start()

    assert third_chunk_started.wait(timeout=1.0)
    release_first_chunk.set()
    consumer.join(timeout=2.0)

    assert not consumer.is_alive()
    assert errors == []
    assert [result.text for result in results] == ["chunk-0", "chunk-1", "chunk-2"]
    assert [result.index for result in results] == [0, 1, 2]


def test_device_normalization() -> None:
    assert normalize_device(" CPU ") == "cpu"
    assert get_runtime_device("rocm") == "cuda"


def test_normalize_device_ids_deduplicates_and_preserves_order() -> None:
    assert normalize_device_ids([2, 0, 2, 1]) == (2, 0, 1)


def test_validate_device_ids_rejects_cpu_runtime() -> None:
    try:
        validate_device_ids("cpu", [0])
        assert False, "Expected device ID validation failure"
    except Exception as exc:  # noqa: BLE001
        assert "WHISPER_DEVICE=cpu" in str(exc)

def test_parse_rocm_smi_output() -> None:
    parsed = _parse_rocm_smi_gpus(
        json.dumps(
            {
                "card0": {
                    "Card Series": "AMD Radeon Pro W6800",
                    "GFX Version": "1100",
                },
                "card1": {
                    "Card Series": "AMD Instinct MI210",
                    "GFX Version": "90a",
                },
            }
        )
    )

    assert [gpu.device_id for gpu in parsed] == [0, 1]
    assert parsed[0].name == "AMD Radeon Pro W6800"
    assert parsed[1].gfx_version == "90a"


def test_parse_rocm_smi_output_without_gfx_version() -> None:
    parsed = _parse_rocm_smi_gpus(
        json.dumps(
            {
                "card0": {
                    "Card Series": "AMD Radeon Pro W6800",
                }
            }
        )
    )

    assert len(parsed) == 1
    assert parsed[0].name == "AMD Radeon Pro W6800"
    assert parsed[0].gfx_version is None


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


def test_list_videos_endpoint(monkeypatch) -> None:
    monkeypatch.setattr("app.main.audio_library_dir", Path("/audios"))
    monkeypatch.setattr(
        "app.main.list_video_library_files",
        lambda _: [
            AudioLibraryFile(filename="sample.mp4", path="sample.mp4", size_bytes=3210),
            AudioLibraryFile(filename="nested.mkv", path="set/nested.mkv", size_bytes=6543),
        ],
    )

    client = TestClient(app)
    response = client.get("/api/videos")
    assert response.status_code == 200
    body = response.json()
    assert body["directory"] == "/audios"
    assert len(body["files"]) == 2
    assert body["files"][0]["url"].endswith("/audios/sample.mp4")
    assert body["files"][1]["url"].endswith("/audios/set/nested.mkv")


def test_list_amd_gpus_endpoint(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.main.detected_amd_gpu_inventory",
        AmdGpuInventory(
            gpus=[
                AmdGpuInfo(
                    device_id=0,
                    name="AMD Instinct MI210",
                    gfx_version="90a",
                )
            ],
            detected_at="2026-04-03T00:00:00+00:00",
            detection_method="rocm-smi",
        ),
    )

    client = TestClient(app)
    response = client.get("/api/hardware/amd-gpus")

    assert response.status_code == 200
    body = response.json()
    assert body["detection_method"] == "rocm-smi"
    assert body["gpus"] == [
        {
            "device_id": 0,
            "name": "AMD Instinct MI210",
            "gfx_version": "90a",
        }
    ]


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
    captured: dict[str, object] = {}

    def fake_transcribe_audio(*args, **kwargs):
        captured["device_ids"] = kwargs.get("device_ids")
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
                "device_ids": [0, 1],
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
        assert captured["device_ids"] == [0, 1]


def test_ws_transcribe_from_audio_library(monkeypatch) -> None:
    called: dict[str, str | None] = {"path": None}

    def fake_resolve_audio_library_file(path: str, *_args):
        called["path"] = path
        return Path("/audios/sample.wav")

    def fake_transcribe_audio(*args, **kwargs):
        yield TranscriptionInfo(duration=5.0)
        yield SegmentResult(index=0, start=0.0, end=5.0, text="from library")

    monkeypatch.setattr("app.main.resolve_audio_library_file", fake_resolve_audio_library_file)
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


def test_ws_v2a_success(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("app.main.audio_library_dir", tmp_path)
    monkeypatch.setattr("app.main.probe_media_duration", lambda _path: 4.0)
    source_video = tmp_path / "set" / "demo.mp4"
    source_video.parent.mkdir(parents=True, exist_ok=True)
    source_video.write_bytes(b"fake video payload")

    def fake_convert_video_to_audio(video_path, output_path, total_duration, cancel_event):
        assert video_path == source_video
        assert total_duration == 4.0
        output_path.write_bytes(b"fake m4a bytes")
        yield VideoConversionProgress(
            processed_seconds=2.0,
            total_estimated_seconds=4.0,
            percent=50.0,
        )
        yield VideoConversionProgress(
            processed_seconds=4.0,
            total_estimated_seconds=4.0,
            percent=100.0,
        )

    monkeypatch.setattr("app.main.convert_video_to_audio", fake_convert_video_to_audio)

    client = TestClient(app)
    with client.websocket_connect("/ws/v2a") as ws:
        ws.send_json(
            {
                "type": "start",
                "video_path": "set/demo.mp4",
                "target_format": "m4a",
            }
        )

        accepted = ws.receive_json()
        progress0 = ws.receive_json()
        progress1 = ws.receive_json()
        progress2 = ws.receive_json()
        complete = ws.receive_json()

        assert accepted["type"] == "accepted"
        assert accepted["filename"] == "demo.mp4"
        assert progress0["type"] == "progress"
        assert progress0["percent"] == 0.0
        assert progress1["percent"] == 50.0
        assert progress2["percent"] == 100.0
        assert complete["type"] == "complete"
        assert complete["filename"] == "demo.m4a"
        assert complete["path"] == "set/demo.m4a"
        assert complete["url"].endswith("/audios/set/demo.m4a")
        assert (tmp_path / "set" / "demo.m4a").read_bytes() == b"fake m4a bytes"


def test_ws_v2a_rejects_invalid_extension(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("app.main.audio_library_dir", tmp_path)
    client = TestClient(app)
    with client.websocket_connect("/ws/v2a") as ws:
        ws.send_json(
            {
                "type": "start",
                "video_path": "demo.mov",
                "target_format": "m4a",
            }
        )

        error = ws.receive_json()
        assert error["type"] == "error"
        assert "Unsupported video extension" in error["message"]


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
