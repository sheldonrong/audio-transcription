import { useState } from "react";

import DisplayPage from "./pages/DisplayPage";
import TranscriptionPage from "./pages/TranscriptionPage";

type Mode = "transcription" | "display";

export default function App() {
  const [mode, setMode] = useState<Mode>("display");
  const [title, setTitle] = useState("");
  const [transcript, setTranscript] = useState("");
  const [segments, setSegments] = useState<Array<{ index: number; start: number; end: number; text: string }>>(
    [],
  );

  return (
    <main className="app-shell">
      <section className="panel">
        <header className="mode-header">
          <h1>Audio Transcription</h1>

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
          </div>
        </header>

        {mode === "transcription" ? (
          <TranscriptionPage
            title={title}
            transcript={transcript}
            onTitleChange={setTitle}
            onTranscriptChange={setTranscript}
            onSegmentsChange={setSegments}
          />
        ) : (
          <DisplayPage title={title} transcript={transcript} segments={segments} />
        )}
      </section>
    </main>
  );
}
