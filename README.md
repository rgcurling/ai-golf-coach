# AI Golf Coach

A browser-based golf swing analyzer that uses MediaPipe Pose (client-side) to extract joint angles from your webcam, compares them frame-by-frame against a reference envelope built from real professional golfer swing videos, and delivers Claude-powered plain-English coaching feedback. This is a portfolio project demonstrating a full computer vision pipeline combined with LLM integration.

## Prerequisites

- Python 3.10+
- Chrome (required for MediaPipe's WASM + GPU delegate)
- An Anthropic API key

## Quick Start

```bash
# 1. Install dependencies and create venv
python setup.py

# 2. Activate the venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Add your API key
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env

# 4. Download pro swing videos (~20-25 clips, 10-60s each)
python fetch_pro_swings.py

# 5. Build the reference envelope from downloaded videos
python build_reference_model.py

# 6. Start the server
python api_server.py

# 7. Open in Chrome
open http://localhost:8080
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  OFFLINE PIPELINE (run once, Python)                            │
│                                                                 │
│  YouTube ──► fetch_pro_swings.py ──► /pro_swings/*.mp4         │
│                       │                                         │
│                       ▼                                         │
│            build_reference_model.py                             │
│            ┌─────────────────────┐                              │
│            │  MediaPipe Pose     │  11 angles × 100 frames     │
│            │  (complexity=2)     │  per swing                   │
│            │  Impact detection   │  normalise to 100 frames    │
│            │  P25/P50/P75 stats  │  impact @ frame 75          │
│            └──────────┬──────────┘                              │
│                       │                                         │
│                       ▼                                         │
│              reference_model.json                               │
└───────────────────────┬─────────────────────────────────────────┘
                        │  served as static file
┌───────────────────────▼─────────────────────────────────────────┐
│  BROWSER (index.html, single file)                              │
│                                                                 │
│  getUserMedia ──► MediaPipe PoseLandmarker (lite, WASM/GPU)    │
│                       │                                         │
│                       │  11 angles per frame                    │
│                       ▼                                         │
│              Normalize to 100 frames                            │
│              Compare vs reference envelope                      │
│              Weighted deviation scoring                         │
│                       │                                         │
│                       ▼                                         │
│              Score + grade + phase timeline                     │
└───────────────────────┬─────────────────────────────────────────┘
                        │  POST /api/feedback
┌───────────────────────▼─────────────────────────────────────────┐
│  api_server.py  (Flask, port 8080)                              │
│                                                                 │
│  Rate limiting (10 req/IP/hr)                                   │
│  ──► Anthropic API (claude-sonnet-4-20250514)                  │
│  ◄── 3 coaching points JSON                                     │
└─────────────────────────────────────────────────────────────────┘
```

## The 11 Features

| Feature | Description |
|---|---|
| `lead_elbow` | Lead arm elbow angle (shoulder→elbow→wrist) |
| `trail_elbow` | Trail arm elbow angle |
| `lead_wrist_angle` | Lead wrist flex proxy |
| `trail_wrist_angle` | Trail wrist flex proxy |
| `lead_knee_flex` | Lead leg knee flexion (hip→knee→ankle) |
| `trail_knee_flex` | Trail leg knee flexion |
| `spine_tilt` | Mid-shoulder to mid-hip angle from vertical |
| `hip_rotation` | Hip open/close from z-depth delta |
| `shoulder_rotation` | Shoulder open/close from z-depth delta |
| `x_factor` | shoulder_rotation − hip_rotation (separation) |
| `hand_height` | Avg wrist Y normalized to shoulder-hip range |

## Improving the Reference Model

The quality of feedback is directly proportional to reference model quality:

1. **More videos**: Edit `MAX_VIDEOS` in `fetch_pro_swings.py` (currently 25)
2. **Better search queries**: Add queries in `SEARCH_QUERIES` — look for face-on AND down-the-line views
3. **Curate manually**: Delete bad files from `/pro_swings` and update `manifest.json`, then re-run `build_reference_model.py`
4. **Add real player data**: If you have access to TrackMan or other biomechanics CSV exports, write a loader that feeds pre-extracted angles directly into the numpy array in `build_reference_model.py`
5. **Camera angle filtering**: Currently mixes face-on and down-the-line. For production, split into two separate envelopes and detect the user's camera angle at recording time

## Project Structure

```
ai-golf-coach/
├── index.html               # Single-file browser app
├── api_server.py            # Flask proxy + static file server
├── fetch_pro_swings.py      # yt-dlp video downloader
├── build_reference_model.py # MediaPipe pipeline + envelope builder
├── setup.py                 # Venv + dependency installer
├── reference_model.json     # Generated — commit after building
├── .env                     # ANTHROPIC_API_KEY (never commit)
├── .env.example             # Template
└── pro_swings/              # Downloaded videos + manifest.json
```
