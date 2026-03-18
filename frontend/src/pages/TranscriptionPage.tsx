import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { TranscriptionWsClient } from "../api/wsClient";
import { getApiBaseUrl, getTranscriptionWsUrl } from "../config";
import type { ServerEvent } from "../types/events";

type UiStatus =
  | "idle"
  | "connecting"
  | "processing"
  | "complete"
  | "aborted"
  | "error";

type TranscribedSegment = {
  index: number;
  start: number;
  end: number;
  text: string;
};

type Props = {
  title: string;
  transcript: string;
  onTitleChange: (value: string) => void;
  onTranscriptChange: (value: string) => void;
  onSegmentsChange: (segments: TranscribedSegment[]) => void;
};

type UploadAudioResponse = {
  detail?: unknown;
  filename?: unknown;
  path?: unknown;
};

const DEFAULT_LANGUAGE = "auto";
const DEFAULT_MODEL = "medium";
const AUDIO_FILE_ACCEPT = "audio/*,.wav,.mp3,.m4a,.mp4,.aac,.ogg,.flac,.webm,.3gp,.amr,.opus,.caf";

function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function uploadAudioFile(
  apiBaseUrl: string,
  file: File,
  onProgress: (percent: number | null) => void,
): Promise<UploadAudioResponse> {
  return new Promise((resolve, reject) => {
    const formData = new FormData();
    formData.append("file", file);

    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${apiBaseUrl}/api/audios/upload`);

    xhr.upload.onprogress = (event) => {
      if (!event.lengthComputable || event.total <= 0) {
        onProgress(null);
        return;
      }
      onProgress((event.loaded / event.total) * 100);
    };

    xhr.onerror = () => {
      reject(new Error("Network error while uploading audio file."));
    };

    xhr.onabort = () => {
      reject(new Error("Audio upload was cancelled."));
    };

    xhr.onload = () => {
      let payload: UploadAudioResponse = {};
      try {
        payload = JSON.parse(xhr.responseText) as UploadAudioResponse;
      } catch {
        payload = {};
      }

      if (xhr.status >= 200 && xhr.status < 300) {
        onProgress(100);
        resolve(payload);
        return;
      }

      const message =
        typeof payload.detail === "string" && payload.detail.trim()
          ? payload.detail
          : `Request failed with status ${xhr.status}.`;
      reject(new Error(message));
    };

    xhr.send(formData);
  });
}

export default function TranscriptionPage({
  title,
  transcript,
  onTitleChange,
  onTranscriptChange,
  onSegmentsChange,
}: Props) {
  const [status, setStatus] = useState<UiStatus>("idle");
  const [audioFiles, setAudioFiles] = useState<
    Array<{ filename: string; path: string; size_bytes: number; url: string }>
  >([]);
  const [selectedAudioPath, setSelectedAudioPath] = useState("");
  const [audioListLoading, setAudioListLoading] = useState(false);
  const [audioListError, setAudioListError] = useState<string | null>(null);
  const [audioUploadLoading, setAudioUploadLoading] = useState(false);
  const [audioUploadProgress, setAudioUploadProgress] = useState<number | null>(null);
  const [audioUploadResult, setAudioUploadResult] = useState<string | null>(null);
  const [transcribedFileMeta, setTranscribedFileMeta] = useState<{
    filename: string;
    path: string;
    libraryPath: string;
  } | null>(null);
  const [progress, setProgress] = useState<number | null>(null);
  const [processedSeconds, setProcessedSeconds] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [exportResult, setExportResult] = useState<string | null>(null);
  const [segments, setSegments] = useState<TranscribedSegment[]>([]);
  const transcriptRef = useRef<HTMLPreElement | null>(null);
  const uploadInputRef = useRef<HTMLInputElement | null>(null);
  const activeClientRef = useRef<TranscriptionWsClient | null>(null);
  const abortRequestedRef = useRef(false);

  const wsUrl = useMemo(() => getTranscriptionWsUrl(), []);
  const apiBaseUrl = useMemo(() => getApiBaseUrl(), []);
  const selectedAudio = useMemo(
    () => audioFiles.find((file) => file.path === selectedAudioPath) ?? null,
    [audioFiles, selectedAudioPath],
  );

  useEffect(() => {
    onSegmentsChange(segments);
  }, [segments, onSegmentsChange]);

  useEffect(() => {
    if (!transcriptRef.current) {
      return;
    }
    transcriptRef.current.scrollTop = transcriptRef.current.scrollHeight;
  }, [transcript]);

  useEffect(() => {
    return () => {
      activeClientRef.current?.close(4000, "Transcription aborted by user");
      activeClientRef.current = null;
    };
  }, []);

  const loadAudioLibrary = useCallback(async () => {
    setAudioListLoading(true);
    setAudioListError(null);

    try {
      const response = await fetch(`${apiBaseUrl}/api/audios`);
      if (!response.ok) {
        throw new Error(`Request failed (${response.status})`);
      }

      const payload = (await response.json()) as {
        files?: Array<{ filename?: unknown; path?: unknown; size_bytes?: unknown; url?: unknown }>;
      };
      const normalized = Array.isArray(payload.files)
        ? payload.files
            .filter((entry) => typeof entry?.filename === "string" && typeof entry?.path === "string")
            .map((entry) => ({
              filename: String(entry.filename),
              path: String(entry.path),
              size_bytes: Number(entry.size_bytes) || 0,
              url: typeof entry.url === "string" && entry.url.trim() ? entry.url : String(entry.path),
            }))
        : [];

      setAudioFiles(normalized);
      setSelectedAudioPath((current) =>
        normalized.some((entry) => entry.path === current) ? current : (normalized[0]?.path ?? ""),
      );
      if (normalized.length === 0) {
        setAudioListError("No audio files found in /audios.");
      }
    } catch (fetchError) {
      const message = fetchError instanceof Error ? fetchError.message : "Failed to load /audios file list.";
      setAudioFiles([]);
      setSelectedAudioPath("");
      setAudioListError(`Failed to load /audios list: ${message}`);
    } finally {
      setAudioListLoading(false);
    }
  }, [apiBaseUrl]);

  useEffect(() => {
    void loadAudioLibrary();
  }, [loadAudioLibrary]);

  const handleAudioUpload = useCallback(
    async (file: File | null) => {
      if (!file) {
        return;
      }

      setAudioUploadLoading(true);
      setAudioUploadProgress(0);
      setAudioUploadResult(null);
      setAudioListError(null);

      try {
        const payload = await uploadAudioFile(apiBaseUrl, file, setAudioUploadProgress);

        const uploadedFilename =
          typeof payload.filename === "string" && payload.filename.trim()
            ? payload.filename
            : file.name;
        const uploadedPath =
          typeof payload.path === "string" && payload.path.trim() ? payload.path : "";

        await loadAudioLibrary();
        if (uploadedPath) {
          setSelectedAudioPath(uploadedPath);
        }
        setAudioUploadResult(`Uploaded audio file: ${uploadedFilename}`);
      } catch (uploadError) {
        setAudioUploadProgress(null);
        const message = uploadError instanceof Error ? uploadError.message : "Failed to upload audio file.";
        setAudioListError(`Upload failed: ${message}`);
      } finally {
        setAudioUploadLoading(false);
      }
    },
    [apiBaseUrl, loadAudioLibrary],
  );

  const handleServerEvent = (event: ServerEvent) => {
    if (abortRequestedRef.current) {
      return;
    }

    if (event.type === "accepted") {
      setStatus("processing");
      return;
    }

    if (event.type === "progress") {
      setStatus("processing");
      setProgress(event.percent);
      setProcessedSeconds(event.processed_seconds);
      return;
    }

    if (event.type === "segment") {
      setStatus("processing");
      setSegments((prev) => {
        const next = [...prev];
        const updated: TranscribedSegment = {
          index: event.index,
          start: event.start,
          end: event.end,
          text: event.text,
        };
        const existingIdx = next.findIndex((seg) => seg.index === event.index);
        if (existingIdx >= 0) {
          next[existingIdx] = updated;
        } else {
          next.push(updated);
          next.sort((a, b) => a.index - b.index);
        }
        return next;
      });
      onTranscriptChange(event.accumulated_text);
      return;
    }

    if (event.type === "complete") {
      setStatus("complete");
      onTranscriptChange(event.text);
      setProgress(100);
      return;
    }

    if (event.type === "error") {
      setStatus("error");
      setError(event.message);
    }
  };

  const handleAbort = () => {
    abortRequestedRef.current = true;
    activeClientRef.current?.close(4000, "Transcription aborted by user");
    activeClientRef.current = null;
    setStatus("aborted");
    setError(null);
  };

  const handleStart = () => {
    if (!title.trim()) {
      setStatus("error");
      setError("Enter a title before starting transcription.");
      return;
    }

    if (!selectedAudio) {
      setStatus("error");
      setError("Choose an audio file from /audios.");
      return;
    }

    activeClientRef.current?.close(4000, "Transcription aborted by user");
    setStatus("connecting");
    abortRequestedRef.current = false;
    setError(null);
    setExportResult(null);
    onTranscriptChange("");
    setSegments([]);
    setTranscribedFileMeta({
      filename: selectedAudio.filename,
      path: selectedAudio.url || selectedAudio.path,
      libraryPath: selectedAudio.path,
    });
    setProgress(null);
    setProcessedSeconds(0);

    const client = new TranscriptionWsClient(wsUrl, {
      onOpen: () => {
        try {
          client.sendStart({
            type: "start",
            filename: selectedAudio.filename,
            audio_path: selectedAudio.path,
            language: DEFAULT_LANGUAGE,
            model: DEFAULT_MODEL,
          });
        } catch (e) {
          setStatus("error");
          setError(`Failed to start transcription: ${(e as Error).message}`);
          client.close();
        }
      },
      onEvent: handleServerEvent,
      onClose: () => {
        activeClientRef.current = null;
        if (abortRequestedRef.current) {
          setStatus("aborted");
          return;
        }
        setStatus((prev) => (prev === "processing" ? "complete" : prev));
      },
      onError: () => {
        if (abortRequestedRef.current) {
          return;
        }
        setStatus("error");
        setError("WebSocket connection failed.");
      },
    });
    activeClientRef.current = client;
  };

  const handleSaveTranscriptionFile = async () => {
    if (segments.length === 0) {
      return;
    }

    const audioMeta =
      transcribedFileMeta ??
      (selectedAudio
        ? {
            filename: selectedAudio.filename,
            path: selectedAudio.url || selectedAudio.path,
            libraryPath: selectedAudio.path,
          }
        : {
            filename: "unknown",
            path: "unknown",
            libraryPath: "",
          });

    if (!audioMeta.libraryPath) {
      setError("Missing audio file path. Start transcription first.");
      return;
    }

    try {
      setError(null);
      setExportResult(null);

      const response = await fetch(`${apiBaseUrl}/api/exports/transcription-file`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          title: title.trim(),
          audio_filename: audioMeta.filename,
          audio_path: audioMeta.libraryPath,
          segments: segments.map((segment) => ({
            start: segment.start,
            end: segment.end,
            text: segment.text,
          })),
        }),
      });

      const data = (await response.json()) as { transcription_file_path?: unknown; detail?: unknown };
      if (!response.ok) {
        const message =
          typeof data.detail === "string" && data.detail.trim()
            ? data.detail
            : `Request failed with status ${response.status}.`;
        throw new Error(message);
      }

      const transcriptionFilePath =
        typeof data.transcription_file_path === "string" ? data.transcription_file_path : "";
      setExportResult(
        transcriptionFilePath
          ? `Transcription File saved to: ${transcriptionFilePath}`
          : "Transcription File saved to OUTPUT_DIR.",
      );
    } catch (exportError) {
      const message = exportError instanceof Error ? exportError.message : "Failed to create Transcription File.";
      setError(`Failed to create Transcription File: ${message}`);
    }
  };

  return (
    <section className="mode-panel">
      <p className="subtitle">Choose audio, stream progress, and watch transcript segments appear live.</p>

      <label className="field title-field">
        <span>Title</span>
        <input
          type="text"
          value={title}
          placeholder="Enter transcript title"
          onChange={(event) => onTitleChange(event.target.value)}
        />
      </label>

      <div className="audio-picker-row">
        <label className="field audio-picker-field" htmlFor="audio-path">
          <span>Audio File (/audios)</span>
          <select
            id="audio-path"
            value={selectedAudioPath}
            disabled={audioListLoading || audioFiles.length === 0}
            onChange={(event) => setSelectedAudioPath(event.target.value)}
          >
            <option value="" disabled>
              {audioListLoading ? "Loading files..." : "Choose an audio file"}
            </option>
            {audioFiles.map((entry) => (
              <option key={entry.path} value={entry.path}>
                {entry.path} ({formatFileSize(entry.size_bytes)})
              </option>
            ))}
          </select>
        </label>
        <button
          type="button"
          className="secondary-btn"
          onClick={() => void loadAudioLibrary()}
          disabled={audioListLoading || audioUploadLoading}
        >
          {audioListLoading ? "Refreshing..." : "Refresh"}
        </button>
        <button
          type="button"
          className="secondary-btn"
          onClick={() => uploadInputRef.current?.click()}
          disabled={audioListLoading || audioUploadLoading}
        >
          {audioUploadLoading ? "Uploading..." : "Upload"}
        </button>
        <input
          ref={uploadInputRef}
          type="file"
          accept={AUDIO_FILE_ACCEPT}
          hidden
          onChange={(event) => {
            const selectedFile = event.currentTarget.files?.[0] ?? null;
            void handleAudioUpload(selectedFile);
            event.currentTarget.value = "";
          }}
        />
      </div>

      {(audioUploadLoading || audioUploadProgress !== null) && (
        <>
          <div className="progress-wrap">
            <div className="progress-bar" style={{ width: `${audioUploadProgress ?? 0}%` }} />
          </div>
          <p className="meta">
            {audioUploadLoading
              ? audioUploadProgress === null
                ? "Uploading audio..."
                : `Upload progress: ${audioUploadProgress.toFixed(1)}%`
              : "Upload complete."}
          </p>
        </>
      )}

      <div className="actions-row">
        <button
          className="start-btn"
          disabled={
            !title.trim() || !selectedAudio || audioListLoading || status === "connecting" || status === "processing"
          }
          onClick={handleStart}
        >
          Start Transcription
        </button>
        <button
          className="secondary-btn"
          disabled={!title.trim() || segments.length === 0}
          onClick={() => void handleSaveTranscriptionFile()}
        >
          Save
        </button>
        <button
          className="secondary-btn"
          disabled={status !== "connecting" && status !== "processing"}
          onClick={handleAbort}
        >
          Stop
        </button>
      </div>

      <div className="status-row">
        <strong>Status:</strong> {status}
        <span className="meta">Processed: {processedSeconds.toFixed(1)}s</span>
      </div>

      <div className="progress-wrap">
        <div className="progress-bar" style={{ width: `${progress ?? 5}%` }} />
      </div>
      <p className="meta">
        {status === "idle"
          ? "Ready to transcribe."
          : status === "aborted"
            ? "Transcription aborted."
            : progress === null
              ? "Processing..."
              : `Progress: ${progress.toFixed(1)}%`}
      </p>

      {error && <p className="error">{error}</p>}
      {exportResult && <p className="meta">{exportResult}</p>}
      {audioUploadResult && <p className="meta">{audioUploadResult}</p>}
      {audioListError && <p className="error">{audioListError}</p>}

      <article className="transcript-box">
        <h2>Transcript</h2>
        <pre ref={transcriptRef}>{transcript || "Transcript will appear here."}</pre>
      </article>
    </section>
  );
}
