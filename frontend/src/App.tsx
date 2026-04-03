import { useEffect, useMemo, useRef, useState } from "react";

import { getApiBaseUrl } from "./config";
import DisplayPage from "./pages/DisplayPage";
import TranscriptionPage from "./pages/TranscriptionPage";
import V2APage from "./pages/V2APage";

type Mode = "transcription" | "display" | "v2a";
type AmdGpu = {
  device_id: number;
  name: string;
  bus_id: string | null;
  uuid: string | null;
};

const GPU_SELECTION_STORAGE_KEY = "dtd:selected-amd-gpu-ids";

function readStoredGpuIds(): number[] {
  try {
    const raw = window.localStorage.getItem(GPU_SELECTION_STORAGE_KEY);
    if (!raw) {
      return [];
    }
    const parsed = JSON.parse(raw) as unknown;
    if (!Array.isArray(parsed)) {
      return [];
    }
    return parsed
      .map((value) => Number(value))
      .filter((value) => Number.isInteger(value) && value >= 0);
  } catch {
    return [];
  }
}

function reconcileSelectedGpuIds(availableGpus: AmdGpu[], preferredIds: number[]): number[] {
  const availableIds = new Set(availableGpus.map((gpu) => gpu.device_id));
  const next = preferredIds.filter((deviceId) => availableIds.has(deviceId));
  if (next.length > 0 || availableGpus.length === 0) {
    return next;
  }
  return availableGpus.map((gpu) => gpu.device_id);
}

export default function App() {
  const [mode, setMode] = useState<Mode>("display");
  const [title, setTitle] = useState("");
  const [transcript, setTranscript] = useState("");
  const [segments, setSegments] = useState<Array<{ index: number; start: number; end: number; text: string }>>(
    [],
  );
  const [amdGpus, setAmdGpus] = useState<AmdGpu[]>([]);
  const [selectedAmdGpuIds, setSelectedAmdGpuIds] = useState<number[]>(() => readStoredGpuIds());
  const [amdGpuLoading, setAmdGpuLoading] = useState(false);
  const [amdGpuError, setAmdGpuError] = useState<string | null>(null);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const settingsRef = useRef<HTMLDivElement | null>(null);
  const apiBaseUrl = useMemo(() => getApiBaseUrl(), []);

  useEffect(() => {
    const handlePointerDown = (event: MouseEvent) => {
      if (!settingsRef.current?.contains(event.target as Node)) {
        setSettingsOpen(false);
      }
    };

    document.addEventListener("mousedown", handlePointerDown);
    return () => document.removeEventListener("mousedown", handlePointerDown);
  }, []);

  useEffect(() => {
    window.localStorage.setItem(GPU_SELECTION_STORAGE_KEY, JSON.stringify(selectedAmdGpuIds));
  }, [selectedAmdGpuIds]);

  useEffect(() => {
    let cancelled = false;

    const loadAmdGpus = async () => {
      setAmdGpuLoading(true);
      setAmdGpuError(null);

      try {
        const response = await fetch(`${apiBaseUrl}/api/hardware/amd-gpus`);
        if (!response.ok) {
          throw new Error(`Request failed (${response.status})`);
        }

        const payload = (await response.json()) as {
          gpus?: Array<{ device_id?: unknown; name?: unknown; bus_id?: unknown; uuid?: unknown }>;
          detection_error?: unknown;
        };
        const normalized = Array.isArray(payload.gpus)
          ? payload.gpus
              .filter(
                (entry) =>
                  Number.isInteger(entry?.device_id) &&
                  typeof entry?.name === "string" &&
                  entry.name.trim().length > 0,
              )
              .map((entry) => ({
                device_id: Number(entry.device_id),
                name: String(entry.name),
                bus_id: typeof entry.bus_id === "string" && entry.bus_id.trim() ? entry.bus_id : null,
                uuid: typeof entry.uuid === "string" && entry.uuid.trim() ? entry.uuid : null,
              }))
          : [];

        if (cancelled) {
          return;
        }

        setAmdGpus(normalized);
        setSelectedAmdGpuIds((current) =>
          reconcileSelectedGpuIds(normalized, current.length > 0 ? current : readStoredGpuIds()),
        );

        const detectionError =
          typeof payload.detection_error === "string" && payload.detection_error.trim()
            ? payload.detection_error
            : null;
        setAmdGpuError(detectionError);
      } catch (error) {
        if (cancelled) {
          return;
        }
        const message = error instanceof Error ? error.message : "Failed to load AMD GPUs.";
        setAmdGpus([]);
        setSelectedAmdGpuIds([]);
        setAmdGpuError(message);
      } finally {
        if (!cancelled) {
          setAmdGpuLoading(false);
        }
      }
    };

    void loadAmdGpus();
    return () => {
      cancelled = true;
    };
  }, [apiBaseUrl]);

  const selectedGpuLabel = useMemo(() => {
    if (amdGpuLoading) {
      return "Detecting GPUs...";
    }
    if (selectedAmdGpuIds.length === 0) {
      return amdGpus.length === 0 ? "No AMD GPUs detected" : "Backend default";
    }
    return `GPU${selectedAmdGpuIds.length > 1 ? "s" : ""} ${selectedAmdGpuIds.join(", ")}`;
  }, [amdGpuLoading, amdGpus.length, selectedAmdGpuIds]);

  const toggleGpuSelection = (deviceId: number) => {
    setSelectedAmdGpuIds((current) =>
      current.includes(deviceId)
        ? current.filter((value) => value !== deviceId)
        : [...current, deviceId].sort((left, right) => left - right),
    );
  };

  return (
    <main className="app-shell">
      <section className="panel">
        <header className="mode-header">
          <div className="mode-header-top">
            <div className="mode-title-wrap">
              <h1>Audio Transcription</h1>
              <p className="header-meta">Inference selection: {selectedGpuLabel}</p>
            </div>

            <div className="settings-wrap" ref={settingsRef}>
              <button
                type="button"
                className={settingsOpen ? "settings-cog active" : "settings-cog"}
                aria-haspopup="dialog"
                aria-expanded={settingsOpen}
                onClick={() => setSettingsOpen((current) => !current)}
              >
                {"\u2699"} Settings
              </button>

              {settingsOpen && (
                <section className="settings-dropdown" aria-label="Inference settings">
                  <div className="settings-copy">
                    <strong>AMD GPU Inference</strong>
                    <p className="meta">Choose the GPU IDs each transcription job should send to the backend.</p>
                  </div>

                  {amdGpuLoading ? (
                    <p className="meta">Detecting available AMD GPUs...</p>
                  ) : amdGpus.length === 0 ? (
                    <p className={amdGpuError ? "error" : "meta"}>
                      {amdGpuError ?? "No AMD GPUs were detected by the backend."}
                    </p>
                  ) : (
                    <div className="settings-list">
                      {amdGpus.map((gpu) => {
                        const checked = selectedAmdGpuIds.includes(gpu.device_id);
                        const details = [gpu.bus_id, gpu.uuid].filter(Boolean).join(" | ");

                        return (
                          <label key={gpu.device_id} className="settings-option">
                            <input
                              type="checkbox"
                              checked={checked}
                              onChange={() => toggleGpuSelection(gpu.device_id)}
                            />
                            <span>
                              <strong>GPU {gpu.device_id}</strong> {gpu.name}
                              {details ? <small>{details}</small> : null}
                            </span>
                          </label>
                        );
                      })}
                    </div>
                  )}

                  {amdGpuError && amdGpus.length > 0 ? <p className="error">{amdGpuError}</p> : null}
                </section>
              )}
            </div>
          </div>

          <div className="mode-switch" role="tablist" aria-label="App mode">
            <button
              type="button"
              role="tab"
              aria-selected={mode === "display"}
              className={mode === "display" ? "mode-btn active" : "mode-btn"}
              onClick={() => setMode("display")}
            >
              Display Mode
            </button>
            <button
              type="button"
              role="tab"
              aria-selected={mode === "transcription"}
              className={mode === "transcription" ? "mode-btn active" : "mode-btn"}
              onClick={() => setMode("transcription")}
            >
              Transcription Mode
            </button>
            <button
              type="button"
              role="tab"
              aria-selected={mode === "v2a"}
              className={mode === "v2a" ? "mode-btn active" : "mode-btn"}
              onClick={() => setMode("v2a")}
            >
              V2A
            </button>
          </div>
        </header>

        {mode === "transcription" ? (
          <TranscriptionPage
            title={title}
            transcript={transcript}
            onTitleChange={setTitle}
            onTranscriptChange={setTranscript}
            onSegmentsChange={setSegments}
            selectedDeviceIds={selectedAmdGpuIds}
          />
        ) : mode === "v2a" ? (
          <V2APage />
        ) : (
          <DisplayPage title={title} transcript={transcript} segments={segments} />
        )}
      </section>
    </main>
  );
}
