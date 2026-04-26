# Contributing

Contributions are welcome — whether that's improving the reference model, refining the angle math, or enhancing the UI.

## Getting Started

```bash
python setup.py
source .venv/bin/activate
cp .env.example .env   # add your ANTHROPIC_API_KEY
python api_server.py
```

## Areas to Contribute

- **Reference model quality** — better video curation, more players, separating face-on vs down-the-line envelopes
- **Angle features** — adding club path, wrist bow/cup, pelvis sway, or other biomechanics metrics
- **Browser app** — comparison overlays, swing history, frame scrubbing
- **Coaching prompts** — more granular Claude system prompts per skill level or swing fault category

## Workflow

1. Fork the repo and create a branch: `git checkout -b feature/your-feature`
2. Make your changes
3. Test in Chrome at `http://localhost:8080`
4. Open a pull request against `main` with a clear description of what changed and why

## Important Notes

- Never commit `.env` or any file containing an API key
- `pose_landmarker_heavy.task` is auto-downloaded — do not commit it
- `pro_swings/*.mp4` are gitignored — include `manifest.json` updates if you change the training set
- The angle math in `index.html` and `build_reference_model.py` must stay in sync — if you change one, change the other

## Reporting Issues

Open an issue with:
- What you expected to happen
- What actually happened
- Browser/OS version and whether you're using webcam or upload
