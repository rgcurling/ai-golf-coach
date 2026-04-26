"""
Local proxy server for the golf swing analyzer.
Serves static files and proxies Claude API calls for coaching feedback.

Usage: python api_server.py
Requires: ANTHROPIC_API_KEY in .env
"""
import json
import os
import time
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import anthropic

load_dotenv()

ROOT = Path(__file__).parent
app = Flask(__name__, static_folder=str(ROOT))
CORS(app)

# Rate limiting: max 10 requests per IP per hour (in-memory, resets on restart)
RATE_LIMIT = 10
RATE_WINDOW = 3600  # seconds
_rate_buckets: dict[str, list[float]] = defaultdict(list)

CLAUDE_MODEL = "claude-sonnet-4-20250514"

SYSTEM_PROMPT = """You are a PGA-certified golf instructor analyzing a student's swing biomechanics data.
You will receive structured data showing how their swing deviates from professional benchmarks.
Respond with exactly 3 coaching points, one per deviation, in this JSON format:
{
  "coaching": [
    { "priority": 1, "headline": "short 4-6 word issue title", "cue": "one specific actionable sentence the golfer can apply immediately" },
    { "priority": 2, "headline": "...", "cue": "..." },
    { "priority": 3, "headline": "...", "cue": "..." }
  ],
  "summary": "one encouraging sentence summarizing overall swing quality"
}"""


def check_rate_limit(ip: str) -> bool:
    """Returns True if request is allowed, False if rate limit exceeded."""
    now = time.time()
    window_start = now - RATE_WINDOW
    _rate_buckets[ip] = [t for t in _rate_buckets[ip] if t > window_start]
    if len(_rate_buckets[ip]) >= RATE_LIMIT:
        return False
    _rate_buckets[ip].append(now)
    return True


@app.route("/")
def serve_index():
    return send_from_directory(str(ROOT), "index.html")


@app.route("/reference_model.json")
def serve_model():
    model_path = ROOT / "reference_model.json"
    if not model_path.exists():
        return jsonify({"error": "reference_model.json not found. Run build_reference_model.py first."}), 404
    return send_from_directory(str(ROOT), "reference_model.json")


@app.route("/api/feedback", methods=["POST"])
def feedback():
    ip = request.remote_addr or "unknown"
    if not check_rate_limit(ip):
        return jsonify({"error": "Rate limit exceeded. Max 10 requests per hour."}), 429

    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key or api_key == "your_key_here":
        return jsonify({"error": "ANTHROPIC_API_KEY not set in .env"}), 500

    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "Empty request body"}), 400

    # Build a clear, structured prompt from the scoring data
    score = data.get("score", 0)
    grade = data.get("grade", "?")
    in_envelope = data.get("in_envelope_pct", 0)
    deviations = data.get("top_deviations", [])

    user_message = f"""Swing analysis results:
- Overall score: {score}/100 (Grade: {grade})
- Frames within pro envelope: {in_envelope:.1f}%

Top deviations from professional benchmarks:
"""
    for i, dev in enumerate(deviations[:3], 1):
        feature = dev.get("feature", "unknown")
        direction = dev.get("direction", "?")
        severity = dev.get("severity", 0)
        mean_delta = dev.get("mean_delta", 0)
        phase = dev.get("phase", "unknown")
        user_message += (
            f"{i}. {feature}: {direction} by avg {mean_delta:.1f}° "
            f"(severity {severity:.1f}) during {phase}\n"
        )

    try:
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=512,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
        raw = message.content[0].text.strip()

        # Strip markdown code fences if Claude wrapped the JSON
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        coaching_json = json.loads(raw)
        return jsonify(coaching_json)

    except json.JSONDecodeError as e:
        return jsonify({"error": f"Claude returned invalid JSON: {e}", "raw": raw}), 502
    except anthropic.APIError as e:
        return jsonify({"error": f"Anthropic API error: {e}"}), 502
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("\n=== Golf Swing Analyzer Server ===")
    print(f"  Serving on: http://localhost:8080")
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key or api_key == "your_key_here":
        print("  [warn] ANTHROPIC_API_KEY not set — coaching feedback will fail")
    model_path = ROOT / "reference_model.json"
    if not model_path.exists():
        print("  [warn] reference_model.json not found — browser will use synthetic fallback")
    print()
    app.run(host="0.0.0.0", port=8080, debug=False)
