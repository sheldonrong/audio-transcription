# FastAPI + React Live Transcription App

This repository now contains a two-part webapp:

- `backend/`: FastAPI WebSocket API using `faster-whisper`
- `frontend/`: React (Vite + TypeScript) UI for live transcription

## Backend setup

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Health check:

```bash
curl http://localhost:8000/health
```

## Frontend setup

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`.

## Docker

Build:

```bash
docker build -t dtd-app .
```

Run:

```bash
docker run --rm -p 8000:8000 \
  -v $(pwd)/audios:/audios \
  -v $(pwd)/output:/output \
  -e WHISPER_DEVICE=cpu \
  dtd-app
```

Open `http://localhost:8000`.

### Docker (AMD ROCm)

Build ROCm image (uses ROCm CTranslate2 fork for faster-whisper):

```bash
docker build -f Dockerfile.rocm -t dtd-app-rocm .
```

Run with ROCm device access:

```bash
docker run --rm -p 8000:8000 \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  -e WHISPER_DEVICE=rocm \
  -v $(pwd)/audios:/audios \
  -v $(pwd)/output:/output \
  dtd-app-rocm
```

Use `HSA_OVERRIDE_GFX_VERSION=9.0.6` instead when your GPU reports `gfx906`.
Optional overrides for legacy Vega cards:

```bash
docker run --rm -p 8000:8000 \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  -e WHISPER_DEVICE=rocm \
  -e HSA_OVERRIDE_GFX_VERSION=9.0.0 \
  -v $(pwd)/audios:/audios \
  -v $(pwd)/output:/output \
  dtd-app-rocm
```

## WebSocket protocol

Endpoint: `ws://localhost:8000/ws/transcribe`

Audio library endpoint:

- `GET http://localhost:8000/api/audios`
- Returns files discovered under `/audios` (or `./audios` at repo root if present)

Export endpoint:

- `POST http://localhost:8000/api/exports/transcription-file`
- Creates a Transcription File (`.tsp`) containing transcript JSON and the selected audio file
- Saves the Transcription File into `OUTPUT_DIR` (default: `./output` in repo root)

Export browsing endpoints (for Display mode Transcription File dropdown):

- `GET http://localhost:8000/api/exports`
- `GET http://localhost:8000/api/exports/{transcription_filename}`
- `GET http://localhost:8000/api/exports/{transcription_filename}/audio?member={archive_member_name}`
- Display mode uses these APIs to populate the Transcription File dropdown and load transcript/audio directly

Client sends:
1. JSON start message:
```json
{"type":"start","filename":"sample.wav","audio_path":"sample.wav","language":"auto","model":"medium"}
```
2. If `audio_path` is omitted: send binary audio payload (single or multiple frames)
3. If uploading bytes: send text frame `"__end__"` to signal upload completion

Server streams events:
- `accepted`
- `progress`
- `segment`
- `complete`
- `error`

## Notes

- Backend defaults: `model=medium`, `device=cuda`, `compute_type=float16`
- `WHISPER_DEVICE` supports `cuda`, `cpu`, and `rocm`
- To run transcription purely on CPU, set `WHISPER_DEVICE=cpu`
- When `WHISPER_DEVICE=cpu` and `WHISPER_COMPUTE_TYPE` is not set, compute type defaults to `int8` for faster CPU transcription
- You can override compute precision explicitly with `WHISPER_COMPUTE_TYPE` (for example `float32`, `int8`, or `int8_float16`)
- `WHISPER_DEVICE=rocm` maps to CTranslate2 GPU runtime and requires the ROCm CTranslate2 build (see `Dockerfile.rocm`)
- You can pass `HSA_OVERRIDE_GFX_VERSION` as an unofficial workaround for legacy cards where ROCm detection fails
- ROCm compatibility is hardware-dependent; current ROCm docs mark `gfx900` unsupported and `gfx906` deprecated, which affects Vega-class GPUs
- Max upload size: `1024 MB`
- You can set `AUDIO_DIR` to override the audio library folder path
- You can set `OUTPUT_DIR` to control where Transcription Files are stored
- Allowed origins include `http://localhost:5173`
- If CUDA is unavailable, backend returns an actionable `error` event.

## Tests

Backend tests:

```bash
cd backend
pytest
```
