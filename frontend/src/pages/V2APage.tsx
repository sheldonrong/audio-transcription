import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { VideoConversionWsClient } from "../api/wsClient";
import { getApiBaseUrl, getV2AWsUrl } from "../config";
import type { VideoConversionServerEvent } from "../types/events";

type UiStatus = "idle" | "connecting" | "converting" | "complete" | "aborted" | "error";

type LibraryVideo = {
  filename: string;
  path: string;
  size_bytes: number;
  url: string;
};

type ConversionResult = {
  filename: string;
  path: string;
  url: string;
};

function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function formatDuration(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds <= 0) {
    return "0:00";
  }

  const rounded = Math.max(0, Math.floor(seconds));
  const minutes = Math.floor(rounded / 60);
  const remainingSeconds = rounded % 60;
  return `${minutes}:${String(remainingSeconds).padStart(2, "0")}`;
}

export default function V2APage() {
  const [status, setStatus] = useState<UiStatus>("idle");
  const [videoDirectory, setVideoDirectory] = useState("/audios");
  const [videoFiles, setVideoFiles] = useState<LibraryVideo[]>([]);
  const [selectedVideoPath, setSelectedVideoPath] = useState("");
  const [videoListLoading, setVideoListLoading] = useState(false);
  const [videoListError, setVideoListError] = useState<string | null>(null);
  const [conversionProgress, setConversionProgress] = useState<number | null>(null);
  const [processedSeconds, setProcessedSeconds] = useState(0);
  const [durationSeconds, setDurationSeconds] = useState<number | null>(null);
  const [result, setResult] = useState<ConversionResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const activeClientRef = useRef<VideoConversionWsClient | null>(null);
  const abortRequestedRef = useRef(false);

  const wsUrl = useMemo(() => getV2AWsUrl(), []);
  const apiBaseUrl = useMemo(() => getApiBaseUrl(), []);
  const selectedVideo = useMemo(
    () => videoFiles.find((entry) => entry.path === selectedVideoPath) ?? null,
    [selectedVideoPath, videoFiles],
  );

  useEffect(() => {
    return () => {
      activeClientRef.current?.close(4000, "Video conversion aborted by user");
      activeClientRef.current = null;
    };
  }, []);

  const loadVideoLibrary = useCallback(async () => {
    setVideoListLoading(true);
    setVideoListError(null);

    try {
      const response = await fetch(`${apiBaseUrl}/api/videos`);
      if (!response.ok) {
        throw new Error(`Request failed (${response.status})`);
      }

      const payload = (await response.json()) as {
        directory?: unknown;
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

      setVideoDirectory(typeof payload.directory === "string" && payload.directory.trim() ? payload.directory : "/audios");
      setVideoFiles(normalized);
      setSelectedVideoPath((current) =>
        normalized.some((entry) => entry.path === current) ? current : (normalized[0]?.path ?? ""),
      );
      if (normalized.length === 0) {
        setVideoListError("No .mkv or .mp4 files found in AUDIO_DIR.");
      }
    } catch (fetchError) {
      const message = fetchError instanceof Error ? fetchError.message : "Failed to load video library.";
      setVideoFiles([]);
      setSelectedVideoPath("");
      setVideoListError(`Failed to load video list: ${message}`);
    } finally {
      setVideoListLoading(false);
    }
  }, [apiBaseUrl]);

  useEffect(() => {
    void loadVideoLibrary();
  }, [loadVideoLibrary]);

  const handleServerEvent = (event: VideoConversionServerEvent) => {
    if (abortRequestedRef.current) {
      return;
    }

    if (event.type === "accepted") {
      setStatus("converting");
      return;
    }

    if (event.type === "progress") {
      setStatus("converting");
      setConversionProgress(event.percent);
      setProcessedSeconds(event.processed_seconds);
      setDurationSeconds(event.total_estimated_seconds);
      return;
    }

    if (event.type === "complete") {
      setStatus("complete");
      setConversionProgress(100);
      setProcessedSeconds((current) => event.duration_seconds ?? current);
      setDurationSeconds(event.duration_seconds);
      setResult({
        filename: event.filename,
        path: event.path,
        url: event.url,
      });
      void loadVideoLibrary();
      return;
    }

    if (event.type === "error") {
      setStatus("error");
      setError(event.message);
    }
  };

  const handleAbort = () => {
    abortRequestedRef.current = true;
    activeClientRef.current?.close(4000, "Video conversion aborted by user");
    activeClientRef.current = null;
    setStatus("aborted");
    setError(null);
  };

  const handleStart = () => {
    if (!selectedVideo) {
      setStatus("error");
      setError("Choose a video from AUDIO_DIR first.");
      return;
    }

    activeClientRef.current?.close(4000, "Video conversion aborted by user");
    abortRequestedRef.current = false;
    setStatus("connecting");
    setError(null);
    setResult(null);
    setConversionProgress(null);
    setProcessedSeconds(0);
    setDurationSeconds(null);

    const client = new VideoConversionWsClient(wsUrl, {
      onOpen: () => {
        try {
          client.sendStart({
            type: "start",
            video_path: selectedVideo.path,
            target_format: "m4a",
          });
        } catch (sendError) {
          setStatus("error");
          setError(`Failed to start conversion: ${(sendError as Error).message}`);
          client.close();
        }
      },
      onEvent: handleServerEvent,
      onClose: () => {
        activeClientRef.current = null;
        if (abortRequestedRef.current) {
          setStatus("aborted");
        }
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

  return (
    <section className="mode-panel">
      <p className="subtitle">Choose a video directly from AUDIO_DIR, convert it to `m4a` with ffmpeg, and save the result back into AUDIO_DIR.</p>

      <div className="audio-picker-row">
        <label className="field audio-picker-field" htmlFor="v2a-video-path">
          <span>Video File ({videoDirectory})</span>
          <select
            id="v2a-video-path"
            value={selectedVideoPath}
            disabled={videoListLoading || videoFiles.length === 0}
            onChange={(event) => setSelectedVideoPath(event.target.value)}
          >
            <option value="" disabled>
              {videoListLoading ? "Loading videos..." : "Choose a video file"}
            </option>
            {videoFiles.map((entry) => (
              <option key={entry.path} value={entry.path}>
                {entry.path} ({formatFileSize(entry.size_bytes)})
              </option>
            ))}
          </select>
        </label>
        <button
          type="button"
          className="secondary-btn"
          onClick={() => void loadVideoLibrary()}
          disabled={videoListLoading || status === "connecting" || status === "converting"}
        >
          {videoListLoading ? "Refreshing..." : "Refresh"}
        </button>
      </div>

      {selectedVideo && (
        <div className="display-meta compact-meta">
          <article className="metric-card">
            <span className="metric-label">Selected Video</span>
            <span className="metric-value metric-value-small">{selectedVideo.path}</span>
          </article>
          <article className="metric-card">
            <span className="metric-label">Size</span>
            <span className="metric-value">{formatFileSize(selectedVideo.size_bytes)}</span>
          </article>
        </div>
      )}

      <div className="actions-row">
        <button
          type="button"
          className="start-btn"
          onClick={handleStart}
          disabled={!selectedVideo || status === "connecting" || status === "converting"}
        >
          {status === "connecting" || status === "converting" ? "Converting..." : "Convert to M4A"}
        </button>
        <button
          type="button"
          className="secondary-btn"
          onClick={handleAbort}
          disabled={status !== "connecting" && status !== "converting"}
        >
          Abort
        </button>
      </div>

      {(status === "connecting" || status === "converting" || status === "complete" || conversionProgress !== null) && (
        <>
          <div className="progress-wrap">
            <div className="progress-bar" style={{ width: `${conversionProgress ?? 0}%` }} />
          </div>
          <p className="meta">
            {status === "connecting" && "Preparing ffmpeg conversion..."}
            {status === "converting" &&
              (conversionProgress === null
                ? `Converting with ffmpeg... ${formatDuration(processedSeconds)} processed`
                : `Converting with ffmpeg: ${conversionProgress.toFixed(1)}% (${formatDuration(processedSeconds)} / ${formatDuration(durationSeconds ?? processedSeconds)})`)}
            {status === "complete" && "Conversion complete."}
          </p>
        </>
      )}

      <div className="status-row">
        <span className="meta">Status: {status}</span>
        {videoListError && <span className="error">{videoListError}</span>}
        {error && <span className="error">{error}</span>}
      </div>

      {result && (
        <div className="transcript-box result-box">
          <h2>Converted Audio</h2>
          <p className="meta">Saved to `{videoDirectory}/{result.path}`.</p>
          <a className="result-link" href={result.url} target="_blank" rel="noreferrer">
            Open generated audio
          </a>
        </div>
      )}
    </section>
  );
}
