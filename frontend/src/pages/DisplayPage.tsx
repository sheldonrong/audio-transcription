import { useCallback, useEffect, useMemo, useRef, useState } from "react";

type Props = {
  title: string;
  transcript: string;
  segments: Array<{ index: number; start: number; end: number; text: string }>;
};

type LoadedSegment = {
  start: number;
  end: number;
  text: string;
};

type LoadedPayload = {
  title?: unknown;
  transcript?: unknown;
  segments?: unknown;
  audio_filename?: unknown;
  audio_path?: unknown;
  audio_url?: unknown;
};

type TranscriptionFileItem = {
  filename: string;
  size_bytes: number;
  modified_at: string;
  url: string;
};

type SegmentView = {
  index: number;
  start: number;
  end: number;
  text: string;
};

type ParagraphView = {
  index: number;
  start: number;
  end: number;
  segments: SegmentView[];
};

function formatSeconds(seconds: number): string {
  return `${seconds.toFixed(2)}s`;
}

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function isLikelyLocalOnlyPath(path: string): boolean {
  const normalized = path.replace(/\\/g, "/");
  const isFakePath = /(^|\/)fakepath\//i.test(normalized);
  const isWindowsAbsolute = /^[a-zA-Z]:\//.test(normalized);
  return isFakePath || isWindowsAbsolute;
}

function groupSegmentsIntoParagraphs(segments: SegmentView[], segmentsPerParagraph: number): ParagraphView[] {
  if (segments.length === 0) return [];

  const paragraphs: ParagraphView[] = [];
  for (let offset = 0, paragraphIndex = 0; offset < segments.length; offset += segmentsPerParagraph, paragraphIndex += 1) {
    const slice = segments.slice(offset, offset + segmentsPerParagraph);
    const start = slice[0]?.start ?? 0;
    const end = slice[slice.length - 1]?.end ?? start;
    paragraphs.push({
      index: paragraphIndex,
      start,
      end,
      segments: slice,
    });
  }
  return paragraphs;
}

function getApiBaseUrl(): string {
  const envUrl = import.meta.env.VITE_API_BASE_URL as string | undefined;
  if (envUrl) return envUrl.replace(/\/+$/, "");
  return "http://localhost:8000";
}

function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function escapeHtml(value: string): string {
  return value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function safeDocName(value: string): string {
  const normalized = value.trim().replace(/\s+/g, "-").replace(/[^a-zA-Z0-9-_]/g, "");
  return normalized || "transcription";
}

export default function DisplayPage({ title, transcript, segments }: Props) {
  const [transcriptionFiles, setTranscriptionFiles] = useState<TranscriptionFileItem[]>([]);
  const [selectedTranscriptionFilename, setSelectedTranscriptionFilename] = useState("");
  const [transcriptionFileListLoading, setTranscriptionFileListLoading] = useState(false);
  const [transcriptionFileDirectory, setTranscriptionFileDirectory] = useState("");
  const [loadedTitle, setLoadedTitle] = useState<string | null>(null);
  const [loadedTranscript, setLoadedTranscript] = useState<string | null>(null);
  const [loadedSegments, setLoadedSegments] = useState<LoadedSegment[]>([]);
  const [loadedAudioMeta, setLoadedAudioMeta] = useState<{ filename: string; path: string } | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [audioError, setAudioError] = useState<string | null>(null);
  const [audioLabel, setAudioLabel] = useState("");
  const [audioSrc, setAudioSrc] = useState<string | null>(null);
  const [segmentsPerParagraph, setSegmentsPerParagraph] = useState(15);
  const [activePlaybackKey, setActivePlaybackKey] = useState<string | null>(null);
  const [playWindow, setPlayWindow] = useState<{ start: number; end: number } | null>(null);
  const [playheadSeconds, setPlayheadSeconds] = useState<number | null>(null);

  const audioRef = useRef<HTMLAudioElement | null>(null);
  const stopTimerRef = useRef<number | null>(null);
  const apiBaseUrl = useMemo(() => getApiBaseUrl(), []);

  const activeTranscript = loadedTranscript ?? transcript;
  const activeTitle = (loadedTitle ?? title).trim() || "Transcription Output";
  const useLoadedData = loadedTranscript !== null;

  const activeSegments = useMemo<SegmentView[]>(() => {
    const base = useLoadedData
      ? loadedSegments.map((segment) => ({ start: segment.start, end: segment.end, text: segment.text }))
      : segments.map((segment) => ({ start: segment.start, end: segment.end, text: segment.text }));
    return base
      .filter((segment) => Number.isFinite(segment.start) && Number.isFinite(segment.end))
      .sort((a, b) => a.start - b.start)
      .map((segment, index) => ({
        index,
        start: segment.start,
        end: segment.end,
        text: segment.text,
      }));
  }, [useLoadedData, loadedSegments, segments]);

  const paragraphViews = useMemo(
    () => groupSegmentsIntoParagraphs(activeSegments, segmentsPerParagraph),
    [activeSegments, segmentsPerParagraph],
  );

  const wordCount = activeTranscript.trim() ? activeTranscript.trim().split(/\s+/).length : 0;

  const clearStopTimer = useCallback(() => {
    if (stopTimerRef.current !== null) {
      window.clearTimeout(stopTimerRef.current);
      stopTimerRef.current = null;
    }
  }, []);

  const stopPlayback = useCallback(() => {
    clearStopTimer();
    if (audioRef.current) {
      audioRef.current.pause();
    }
    setActivePlaybackKey(null);
    setPlayWindow(null);
    setPlayheadSeconds(null);
  }, [clearStopTimer]);

  useEffect(() => {
    if (!audioSrc) {
      return;
    }

    stopPlayback();
    if (!audioRef.current) {
      audioRef.current = new Audio();
    }

    const audio = audioRef.current;
    const handleTimeUpdate = () => {
      setPlayheadSeconds(audio.currentTime);
    };
    const handleEnded = () => {
      stopPlayback();
    };

    audio.src = audioSrc;
    audio.preload = "auto";
    audio.load();
    audio.addEventListener("timeupdate", handleTimeUpdate);
    audio.addEventListener("ended", handleEnded);

    return () => {
      audio.removeEventListener("timeupdate", handleTimeUpdate);
      audio.removeEventListener("ended", handleEnded);
    };
  }, [audioSrc, stopPlayback]);

  useEffect(() => {
    return () => {
      stopPlayback();
    };
  }, [stopPlayback]);

  const loadTranscriptionFileList = useCallback(async () => {
    setTranscriptionFileListLoading(true);

    try {
      const response = await fetch(`${apiBaseUrl}/api/exports`);
      const body = (await response.json()) as {
        directory?: unknown;
        files?: Array<{ filename?: unknown; size_bytes?: unknown; modified_at?: unknown; url?: unknown }>;
        detail?: unknown;
      };
      if (!response.ok) {
        const message =
          typeof body.detail === "string" && body.detail.trim()
            ? body.detail
            : `Request failed (${response.status})`;
        throw new Error(message);
      }

      const files = Array.isArray(body.files)
        ? body.files
            .filter((item) => typeof item?.filename === "string")
            .map((item) => ({
              filename: String(item.filename),
              size_bytes: Number(item.size_bytes) || 0,
              modified_at: typeof item.modified_at === "string" ? item.modified_at : "",
              url: typeof item.url === "string" ? item.url : "",
            }))
        : [];

      setTranscriptionFiles(files);
      setTranscriptionFileDirectory(typeof body.directory === "string" ? body.directory : "");
      setSelectedTranscriptionFilename((current) =>
        files.some((item) => item.filename === current) ? current : (files[0]?.filename ?? ""),
      );
      if (files.length === 0) {
        setLoadError("No Transcription Files found in OUTPUT_DIR.");
      } else {
        setLoadError(null);
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : "Failed to load Transcription Files.";
      setTranscriptionFiles([]);
      setSelectedTranscriptionFilename("");
      setLoadError(`Failed to load Transcription File list: ${message}`);
    } finally {
      setTranscriptionFileListLoading(false);
    }
  }, [apiBaseUrl]);

  useEffect(() => {
    void loadTranscriptionFileList();
  }, [loadTranscriptionFileList]);

  const playSegment = useCallback(
    (startTime: number, endTime: number, playbackKey: string) => {
      if (!audioRef.current || !audioSrc) {
        setAudioError("Load an audio file first to play segments.");
        return;
      }

      if (endTime <= startTime) {
        return;
      }

      const start = Math.max(0, startTime);
      const end = Math.max(start, endTime);
      const durationMs = Math.max(0, (end - start) * 1000);
      const audio = audioRef.current;

      const startPlayback = async () => {
        stopPlayback();

        try {
          if (audio.readyState < 1) {
            await new Promise<void>((resolve, reject) => {
              const onLoaded = () => {
                cleanup();
                resolve();
              };
              const onError = () => {
                cleanup();
                reject(new Error("Unable to load audio metadata."));
              };
              const cleanup = () => {
                audio.removeEventListener("loadedmetadata", onLoaded);
                audio.removeEventListener("error", onError);
              };

              audio.addEventListener("loadedmetadata", onLoaded);
              audio.addEventListener("error", onError);
              audio.load();
            });
          }

          audio.currentTime = start;
          await new Promise<void>((resolve) => {
            if (Math.abs(audio.currentTime - start) <= 0.05) {
              resolve();
              return;
            }

            const onSeeked = () => {
              cleanup();
              resolve();
            };
            const onError = () => {
              cleanup();
              resolve();
            };
            const timeoutId = window.setTimeout(() => {
              cleanup();
              resolve();
            }, 600);
            const cleanup = () => {
              window.clearTimeout(timeoutId);
              audio.removeEventListener("seeked", onSeeked);
              audio.removeEventListener("error", onError);
            };

            audio.addEventListener("seeked", onSeeked);
            audio.addEventListener("error", onError);
          });

          setActivePlaybackKey(playbackKey);
          setPlayWindow({ start, end });
          setPlayheadSeconds(start);
          await audio.play();

          stopTimerRef.current = window.setTimeout(() => {
            stopPlayback();
          }, durationMs);
          setAudioError(null);
        } catch (error: unknown) {
          const message = error instanceof Error ? error.message : "Unable to play audio.";
          setAudioError(message);
          stopPlayback();
        }
      };

      void startPlayback();
    },
    [audioSrc, stopPlayback],
  );

  const currentPlayingSegment = useMemo(() => {
    if (!playWindow || playheadSeconds === null) {
      return null;
    }

    const epsilon = 0.03;
    const t = playheadSeconds;
    if (t < playWindow.start - epsilon || t > playWindow.end + epsilon) {
      return null;
    }

    return (
      activeSegments.find(
        (segment) =>
          t >= segment.start - epsilon &&
          t <= segment.end + epsilon &&
          segment.end >= playWindow.start - epsilon &&
          segment.start <= playWindow.end + epsilon,
      ) ?? null
    );
  }, [activeSegments, playWindow, playheadSeconds]);

  const handleLoadTranscriptionFile = async () => {
    if (!selectedTranscriptionFilename) {
      return;
    }

    try {
      const response = await fetch(
        `${apiBaseUrl}/api/exports/${encodeURIComponent(selectedTranscriptionFilename)}`,
      );
      const parsed = (await response.json()) as LoadedPayload & { detail?: unknown };
      if (!response.ok) {
        const message =
          typeof parsed.detail === "string" && parsed.detail.trim()
            ? parsed.detail
            : `Request failed (${response.status})`;
        throw new Error(message);
      }

      if (!isObject(parsed)) {
        throw new Error("Transcription File payload must be an object.");
      }

      const parsedSegments = Array.isArray(parsed.segments)
        ? parsed.segments
            .filter((segment): segment is Record<string, unknown> => isObject(segment))
            .map((segment) => ({
              start: Number(segment.start),
              end: Number(segment.end),
              text: typeof segment.text === "string" ? segment.text : "",
            }))
            .filter((segment) => Number.isFinite(segment.start) && Number.isFinite(segment.end))
            .sort((a, b) => a.start - b.start)
        : [];

      const transcriptFromJson =
        typeof parsed.transcript === "string"
          ? parsed.transcript
          : parsedSegments.map((segment) => segment.text).join(" ").trim();
      const titleFromJson = typeof parsed.title === "string" ? parsed.title.trim() : "";

      const audioFilename =
        typeof parsed.audio_filename === "string" && parsed.audio_filename.trim()
          ? parsed.audio_filename
          : selectedTranscriptionFilename;
      const audioPath =
        typeof parsed.audio_path === "string" && parsed.audio_path.trim()
          ? parsed.audio_path
          : audioFilename;
      const audioUrl =
        typeof parsed.audio_url === "string" && parsed.audio_url.trim()
          ? parsed.audio_url
          : "";

      if (!transcriptFromJson && parsedSegments.length === 0) {
        throw new Error("Transcription File transcript JSON must include `transcript` or `segments`.");
      }

      setLoadedTranscript(transcriptFromJson);
      setLoadedTitle(titleFromJson || null);
      setLoadedSegments(parsedSegments);
      setLoadedAudioMeta({ filename: audioFilename, path: audioPath });

      if (audioUrl) {
        setAudioSrc(audioUrl);
        setAudioLabel(audioFilename);
        setAudioError(null);
      } else if (audioPath && !isLikelyLocalOnlyPath(audioPath)) {
        const normalizedPath = audioPath.replace(/\\/g, "/");
        setAudioSrc(normalizedPath);
        setAudioLabel(audioFilename);
        setAudioError(null);
      } else {
        setAudioError("Audio in this Transcription File is not directly playable.");
      }

      setLoadError(null);
    } catch (error) {
      const message = error instanceof Error ? error.message : "Invalid Transcription File.";
      setLoadError(`Failed to load Transcription File: ${message}`);
    }
  };

  const handleDownloadWord = () => {
    const paragraphs =
      paragraphViews.length > 0
        ? paragraphViews
            .map((paragraph) => paragraph.segments.map((segment) => segment.text).join(" ").trim())
            .filter((text) => text.length > 0)
        : activeTranscript.trim()
          ? [activeTranscript.trim()]
          : [];

    if (paragraphs.length === 0) {
      return;
    }

    const titleText = activeTitle.trim() || "Transcription Output";
    const html = `<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8" />
  <title>${escapeHtml(titleText)}</title>
  <style>
    body { font-family: Calibri, Arial, sans-serif; line-height: 1.5; margin: 24px; }
    h1 { margin: 0 0 16px; font-size: 22px; }
    p { margin: 0 0 12px; }
  </style>
</head>
<body>
  <h1>${escapeHtml(titleText)}</h1>
  ${paragraphs.map((text) => `<p>${escapeHtml(text)}</p>`).join("\n  ")}
</body>
</html>`;

    const blob = new Blob([html], { type: "application/msword;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `${safeDocName(titleText)}.doc`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 0);
  };

  return (
    <section className="mode-panel">
      <p className="subtitle">Formatted presentation view for completed transcription output.</p>

      <div className="audio-picker-row">
        <label className="field audio-picker-field" htmlFor="output-transcription-file">
          <span>Transcription File</span>
          <select
            id="output-transcription-file"
            value={selectedTranscriptionFilename}
            disabled={transcriptionFileListLoading || transcriptionFiles.length === 0}
            onChange={(event) => setSelectedTranscriptionFilename(event.target.value)}
          >
            <option value="" disabled>
              {transcriptionFileListLoading
                ? "Loading Transcription Files..."
                : "Choose a Transcription File"}
            </option>
            {transcriptionFiles.map((item) => (
              <option key={item.filename} value={item.filename}>
                {item.filename} ({formatFileSize(item.size_bytes)})
              </option>
            ))}
          </select>
        </label>
        <button
          type="button"
          className="secondary-btn"
          onClick={() => void loadTranscriptionFileList()}
          disabled={transcriptionFileListLoading}
        >
          {transcriptionFileListLoading ? "Refreshing..." : "Refresh"}
        </button>
        <button
          type="button"
          className="secondary-btn"
          onClick={() => void handleLoadTranscriptionFile()}
          disabled={!selectedTranscriptionFilename || transcriptionFileListLoading}
        >
          Load Transcription File
        </button>
        <button
          type="button"
          className="secondary-btn"
          onClick={handleDownloadWord}
          disabled={paragraphViews.length === 0 && !activeTranscript.trim()}
        >
          Download Word
        </button>
      </div>
      {loadError && <p className="error">{loadError}</p>}
      {audioError && <p className="error">{audioError}</p>}
      {playWindow && (
        <p className="meta">
          Playing {formatSeconds(playWindow.start)} - {formatSeconds(playWindow.end)}
          {currentPlayingSegment
            ? ` | Segment ${currentPlayingSegment.index + 1}: ${formatSeconds(currentPlayingSegment.start)} - ${formatSeconds(currentPlayingSegment.end)}`
            : ""}
        </p>
      )}
      <div className="slider-row">
        <label htmlFor="segments-per-paragraph">Paragraph break every {segmentsPerParagraph} segments</label>
        <input
          id="segments-per-paragraph"
          className="range-input"
          type="range"
          min={9}
          max={25}
          step={1}
          value={segmentsPerParagraph}
          onChange={(event) => setSegmentsPerParagraph(Number(event.target.value))}
        />
      </div>

      <article className="display-paper">
        <h2>{activeTitle}</h2>
        {paragraphViews.length === 0 ? (
          <p className="meta">{activeTranscript || "No transcript yet. Run transcription mode first, then switch back here."}</p>
        ) : (
          paragraphViews.map((paragraph) => {
            const paragraphKey = `paragraph-${paragraph.index}`;
            const isParagraphPlaying =
              activePlaybackKey === paragraphKey ||
              activePlaybackKey?.startsWith(`segment-${paragraph.index}-`) === true;

            return (
              <div className="paragraph-row" key={paragraphKey}>
                <button
                  type="button"
                  className="play-btn"
                  data-start-time={paragraph.start.toFixed(3)}
                  data-end-time={paragraph.end.toFixed(3)}
                  title={`Start: ${formatSeconds(paragraph.start)} | End: ${formatSeconds(paragraph.end)}`}
                  aria-label={isParagraphPlaying ? "Stop playback" : "Play paragraph"}
                  onClick={() => {
                    if (isParagraphPlaying) {
                      stopPlayback();
                    } else {
                      playSegment(paragraph.start, paragraph.end, paragraphKey);
                    }
                  }}
                >
                  {isParagraphPlaying ? "◼" : "▶"}
                </button>
                <p className="display-transcript">
                  {paragraph.segments.map((segment, segmentIndex) => {
                    const segmentKey = `segment-${paragraph.index}-${segmentIndex}`;
                    const isSegmentPlaying = activePlaybackKey === segmentKey;
                    const isCurrentlyPlaying = currentPlayingSegment?.index === segment.index;
                    const hoverLabel = `Start: ${formatSeconds(segment.start)} | End: ${formatSeconds(segment.end)}`;
                    const className = [
                      "hover-sentence",
                      "timed",
                      isCurrentlyPlaying ? "playing-segment" : "",
                      isSegmentPlaying ? "selected-segment" : "",
                    ]
                      .filter(Boolean)
                      .join(" ");

                    return (
                      <span
                        key={segmentKey}
                        className={className}
                        title={hoverLabel}
                        data-time={hoverLabel}
                        data-start-time={segment.start.toFixed(3)}
                        data-end-time={segment.end.toFixed(3)}
                        role="button"
                        tabIndex={0}
                        onClick={() => {
                          if (isSegmentPlaying) {
                            stopPlayback();
                          } else {
                            playSegment(segment.start, segment.end, segmentKey);
                          }
                        }}
                        onKeyDown={(event) => {
                          if (event.key === "Enter" || event.key === " ") {
                            event.preventDefault();
                            if (isSegmentPlaying) {
                              stopPlayback();
                            } else {
                              playSegment(segment.start, segment.end, segmentKey);
                            }
                          }
                        }}
                      >
                        {segment.text}
                        {segmentIndex < paragraph.segments.length - 1 ? " " : ""}
                      </span>
                    );
                  })}
                </p>
              </div>
            );
          })
        )}
      </article>
    </section>
  );
}
