# Changelog

## v1.0.0 — 2026-04-26

Initial release of the AI Golf Coach pipeline.

### Added
- `fetch_pro_swings.py` — automated yt-dlp downloader targeting 9 specific PGA Tour players across face-on and down-the-line camera angles
- `build_reference_model.py` — MediaPipe Tasks pose pipeline that extracts 11 joint angle features per frame, detects impact via wrist velocity peak, normalises each swing to 100 frames, and computes P25/P50/P75 envelopes with feature importance weights
- `index.html` — single-file browser app with real-time MediaPipe skeleton overlay, 10-second swing recorder, video upload fallback, animated score ring, phase timeline bar, and Claude-powered coaching cards
- `api_server.py` — Flask server serving the app and proxying Claude API calls with per-IP rate limiting
- `setup.py` — one-command venv + dependency installer
- `reference_model.json` — pre-built envelope from 32 professional swing videos (Rory McIlroy, Jon Rahm, Scottie Scheffler, Justin Thomas, Jordan Spieth, and others)
