# AI Golf Coach

A browser-based golf swing analyzer that uses MediaPipe Pose (client-side) to extract joint angles from your webcam, compares them frame-by-frame against per-club reference envelopes built from real professional golfer swing videos, and delivers Claude-powered plain-English coaching feedback.

Built as a portfolio project demonstrating a full computer vision pipeline combined with LLM integration, a Supabase backend, and a Progressive Web App frontend.

---

## Prerequisites

- Python 3.10+
- Chrome (required for MediaPipe's WASM + GPU delegate)
- An [Anthropic API key](https://console.anthropic.com/)
- (Optional) A [Supabase](https://supabase.com/) project for swing storage and caching

---

## Quick Start

```bash
# 1. Install dependencies
python setup.py
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2. Configure environment
cp .env.example .env
# Edit .env and add ANTHROPIC_API_KEY (and optionally SUPABASE_URL / SUPABASE_KEY)

# 3. Download pro swing videos for a club
python fetch_pro_swings.py --club driver
python fetch_pro_swings.py --club mid_iron

# 4. Build per-club reference models
python build_reference_model.py --club driver
python build_reference_model.py --club mid_iron

# 5. Start the server
python api_server.py

# 6. Open in Chrome
open http://localhost:8080
```

---

## Building All Six Club Models

The app supports six club categories, each with its own reference envelope and scoring weights:

| Club | Download target | Fallback if not built |
|---|---|---|
| `driver` | 40 videos | — |
| `fairway_wood` | 25 videos | driver |
| `long_iron` | 25 videos | mid_iron |
| `mid_iron` | 40 videos | — |
| `short_iron` | 25 videos | mid_iron |
| `wedge` | 30 videos | combined |

**Recommended build order** (by impact on coaching quality):

```bash
# Core two — build these first
python fetch_pro_swings.py --club driver   && python build_reference_model.py --club driver
python fetch_pro_swings.py --club mid_iron && python build_reference_model.py --club mid_iron

# High value — most mechanically distinct
python fetch_pro_swings.py --club wedge      && python build_reference_model.py --club wedge
python fetch_pro_swings.py --club short_iron && python build_reference_model.py --club short_iron

# Lower priority — decent fallbacks already in place
python fetch_pro_swings.py --club fairway_wood && python build_reference_model.py --club fairway_wood
python fetch_pro_swings.py --club long_iron    && python build_reference_model.py --club long_iron

# Or build everything at once
python fetch_pro_swings.py --club all
python build_reference_model.py --club all
```

Models are saved to `reference_models/{club}.json`. The server loads them all at startup and falls back gracefully when a club model hasn't been built yet.

---

## Supabase Setup (Optional)

Supabase adds swing history, response caching, and per-IP daily rate limiting. The app works without it — all Supabase calls fail silently.

**1. Create a project at [supabase.com](https://supabase.com) and run `setup_supabase.sql` in the SQL Editor.**

**2. Add credentials to `.env`:**
```
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-anon-key
```

**3. Verify the connection:**
```bash
python test_supabase.py
```

Features enabled with Supabase:
- **Swing storage** — every analyzed swing is stored with score, deviations, and coaching feedback
- **Response cache** — identical deviation patterns return cached coaching without a Claude API call
- **Daily limit** — 5 free analyses per IP per day (configurable via `FREE_DAILY_LIMIT` in `api_server.py`)
- **Stats endpoint** — `GET /api/stats` returns total swings, cached patterns, and cache hit count

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  OFFLINE PIPELINE (run once per club, Python)                   │
│                                                                 │
│  YouTube ──► fetch_pro_swings.py ──► pro_swings/{club}/*.mp4  │
│                       │                                         │
│                       ▼                                         │
│            build_reference_model.py --club {club}               │
│            ┌─────────────────────┐                              │
│            │  MediaPipe Pose     │  12 features × 100 frames   │
│            │  Impact detection   │  normalised, impact @ f75   │
│            │  P25/P50/P75 stats  │  + tempo ratio stats        │
│            │  Per-club weights   │                              │
│            └──────────┬──────────┘                              │
│                       ▼                                         │
│              reference_models/{club}.json                       │
└───────────────────────┬─────────────────────────────────────────┘
                        │  served as static files
┌───────────────────────▼─────────────────────────────────────────┐
│  BROWSER (index.html, single file)                              │
│                                                                 │
│  6-club selector ──► getUserMedia / video upload               │
│                       │                                         │
│                       │  MediaPipe PoseLandmarker (lite/WASM)  │
│                       │  12 features per frame                  │
│                       ▼                                         │
│              Normalize to 100 frames (impact @ 75)             │
│              Compare vs per-club reference envelope             │
│              Weighted deviation scoring + tempo ratio           │
│                       │                                         │
│                       ▼                                         │
│              Score + grade + radar chart + phase timeline       │
└───────────────────────┬─────────────────────────────────────────┘
                        │  POST /api/feedback  {club_category, tempo_ratio, ...}
┌───────────────────────▼─────────────────────────────────────────┐
│  api_server.py  (Flask, port 8080)                              │
│                                                                 │
│  In-memory rate limit (10 req/IP/hr)                           │
│  Supabase daily limit (5 analyses/IP/day)                      │
│  Supabase response cache (MD5 of deviation pattern + club)     │
│  ──► Anthropic API (claude-sonnet-4-20250514)                  │
│  ◄── 3 coaching points + optional tempo cue JSON               │
│  Supabase swing storage                                         │
└─────────────────────────────────────────────────────────────────┘
```

---

## The 12 Features

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
| `tempo_ratio` | Backswing frames / downswing frames (constant per swing) |

Feature weights are tuned per club — `x_factor` and `hip_rotation` are weighted highest for driver, `spine_tilt` and `trail_wrist_angle` for wedge.

---

## Per-Club Scoring Weights

Each model blends consistency-based weights (features more consistent across pros score higher) with club-specific importance weights:

| Feature | Driver | Mid Iron | Wedge |
|---|---|---|---|
| x_factor | 2.0 | 1.6 | 1.2 |
| spine_tilt | 1.4 | 1.8 | 2.0 |
| hip_rotation | 1.8 | 1.4 | 1.3 |
| trail_wrist_angle | 1.0 | 1.3 | 1.8 |
| tempo_ratio | 1.5 | 1.4 | 1.6 |

---

## Coaching Cache

Pre-generate 396 coaching cues (6 clubs × 11 features × 2 directions × 3 severities) using Claude Haiku to reduce live API costs:

```bash
python generate_coaching_cache.py
```

Saves to `app/coaching_cache.json`. Resumes from checkpoint if interrupted. Estimated cost: ~$0.012 for the full set.

---

## Deployment (Railway)

```bash
# Set environment variables in Railway dashboard:
# ANTHROPIC_API_KEY, SUPABASE_URL, SUPABASE_KEY

# Deploy
railway up
```

The `Procfile` runs gunicorn with a single worker. Railway provides HTTPS automatically, which is required for `getUserMedia` on mobile browsers.

---

## Project Structure

```
ai-golf-coach/
├── index.html                    # Single-file browser app (MediaPipe + UI)
├── api_server.py                 # Flask server, Claude proxy, Supabase integration
├── fetch_pro_swings.py           # yt-dlp downloader — python fetch_pro_swings.py --club <club>
├── build_reference_model.py      # MediaPipe pipeline — python build_reference_model.py --club <club>
├── generate_coaching_cache.py    # Pre-generate 396 coaching cues via Claude Haiku
├── setup.py                      # Venv + dependency installer
├── setup_supabase.sql            # Database schema (run once in Supabase SQL Editor)
├── test_supabase.py              # Connection verification
├── inspect_model.py              # Debug/analyse a built reference model
├── manifest.json                 # PWA manifest
├── Procfile                      # gunicorn config for Railway
├── requirements.txt              # Python dependencies
├── .env.example                  # Environment variable template
├── reference_models/             # Built per-club envelopes (committed)
│   ├── driver.json
│   ├── mid_iron.json
│   ├── wedge.json
│   └── short_iron.json
├── pro_swings/                   # Downloaded videos (not committed) + manifests
│   ├── driver/manifest.json
│   ├── mid_iron/manifest.json
│   ├── wedge/manifest.json
│   └── short_iron/manifest.json
└── app/
    └── coaching_cache.json       # Generated coaching cues (optional)
```

---

## Improving Model Quality

1. **More videos** — increase `target_videos` in `CLUB_CONFIGS` inside `fetch_pro_swings.py`
2. **Better queries** — add more specific player/angle queries to `CLUB_CONFIGS`
3. **Curate manually** — delete low-quality files from `pro_swings/{club}/`, update `manifest.json`, rebuild
4. **Camera angle filtering** — currently mixes face-on and down-the-line views; splitting into separate envelopes and detecting the user's angle at record time would improve accuracy
5. **Real biomechanics data** — if you have TrackMan or force-plate CSV exports, write a loader that feeds pre-extracted angles directly into the numpy pipeline in `build_reference_model.py`
