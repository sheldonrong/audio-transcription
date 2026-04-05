from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class StartMessage(BaseModel):
    type: Literal["start"]
    filename: str = Field(min_length=1)
    audio_path: Optional[str] = None
    language: Optional[str] = "en"
    model: Optional[str] = "medium"
    device_ids: list[int] = Field(default_factory=list)


class AmdGpuInfoResponse(BaseModel):
    device_id: int = Field(ge=0)
    name: str = Field(min_length=1)
    gfx_version: Optional[str] = None


class AmdGpuListResponse(BaseModel):
    gpus: list[AmdGpuInfoResponse]
    detected_at: str
    detection_method: Optional[str] = None
    detection_error: Optional[str] = None


class VideoConversionStartMessage(BaseModel):
    type: Literal["start"]
    video_path: str = Field(min_length=1)
    target_format: Literal["m4a"] = "m4a"


class ExportSegment(BaseModel):
    start: float
    end: float
    text: str


class ExportTranscriptionFileRequest(BaseModel):
    title: str = Field(min_length=1)
    audio_filename: str = Field(min_length=1)
    audio_path: str = Field(min_length=1)
    segments: list[ExportSegment] = Field(min_length=1)


class ExportTranscriptionFileResponse(BaseModel):
    transcription_file_filename: str
    transcription_file_path: str
    json_filename: str
    audio_filename: str


class ExportTranscriptionFileSummary(BaseModel):
    filename: str
    size_bytes: int
    modified_at: str
    url: str


class ExportTranscriptionFileListResponse(BaseModel):
    directory: str
    files: list[ExportTranscriptionFileSummary]


class ExportTranscriptionFileLoadResponse(BaseModel):
    transcription_file_filename: str
    title: str
    transcript: str
    segments: list[ExportSegment]
    audio_filename: str
    audio_path: str
    audio_url: str


class AcceptedEvent(BaseModel):
    type: Literal["accepted"] = "accepted"
    job_id: str
    filename: str


class ProgressEvent(BaseModel):
    type: Literal["progress"] = "progress"
    job_id: str
    percent: Optional[float]
    processed_seconds: float
    total_estimated_seconds: Optional[float]


class SegmentEvent(BaseModel):
    type: Literal["segment"] = "segment"
    job_id: str
    index: int
    start: float
    end: float
    text: str
    accumulated_text: str


class CompleteEvent(BaseModel):
    type: Literal["complete"] = "complete"
    job_id: str
    text: str
    segments_count: int
    duration_seconds: Optional[float]


class ErrorEvent(BaseModel):
    type: Literal["error"] = "error"
    job_id: str
    message: str


class VideoConversionProgressEvent(BaseModel):
    type: Literal["progress"] = "progress"
    job_id: str
    percent: Optional[float]
    processed_seconds: float
    total_estimated_seconds: Optional[float]


class VideoConversionCompleteEvent(BaseModel):
    type: Literal["complete"] = "complete"
    job_id: str
    filename: str
    path: str
    url: str
    duration_seconds: Optional[float]
