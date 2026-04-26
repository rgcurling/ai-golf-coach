"""
Local proxy server for the golf swing analyzer.
Serves static files and proxies Claude API calls for coaching feedback.

Usage: python api_server.py
Requires: ANTHROPIC_API_KEY in .env
"""
import hashlib
import json
import os
import time
from collections import defaultdict
from datetime import date
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import anthropic
from supabase import create_client

load_dotenv()

ROOT = Path(__file__).parent
app = Flask(__name__, static_folder=str(ROOT))
CORS(app)

RATE_LIMIT  = 10
RATE_WINDOW = 3600
_rate_buckets: dict[str, list[float]] = defaultdict(list)

FREE_DAILY_LIMIT = 5

_supabase = None
try:
    _supabase = create_client(os.getenv("SUPABASE_URL", ""), os.getenv("SUPABASE_KEY", ""))
except Exception as _e:
    print(f"  [warn] Supabase init failed: {_e}")

CLAUDE_MODEL = "claude-sonnet-4-20250514"

SYSTEM_PROMPT = """You are a PGA-certified golf instructor analyzing a student's swing biomechanics data.
You will receive structured data showing how their swing deviates from professional benchmarks for a specific club.
Tailor your advice to the club being used — driver swings require more rotation and wider arc, iron swings require steeper attack angle and shaft lean.
Respond with exactly 3 coaching points, one per deviation, in this JSON format:
{
  "coaching": [
    { "priority": 1, "headline": "short 4-6 word issue title", "cue": "one specific actionable sentence the golfer can apply immediately" },
    { "priority": 2, "headline": "...", "cue": "..." },
    { "priority": 3, "headline": "...", "cue": "..." }
  ],
  "summary": "one encouraging sentence summarizing overall swing quality"
}"""


def make_signature(deviations: list) -> str:
    sig = json.dumps([
        {"feature": d["feature"], "direction": d["direction"], "severity_bucket": round(d.get("severity", 0))}
        for d in sorted(deviations[:3], key=lambda x: x["feature"])
    ], sort_keys=True)
    return hashlib.md5(sig.encode()).hexdigest()


def check_supabase_cache(signature: str):
    if not _supabase:
        return None
    try:
        result = _supabase.table("coaching_cache").select("response, hit_count").eq("signature", signature).execute()
        if result.data:
            _supabase.table("coaching_cache").update({"hit_count": result.data[0]["hit_count"] + 1}).eq("signature", signature).execute()
            return result.data[0]["response"]
    except Exception as e:
        print(f"Cache check error: {e}")
    return None


def store_supabase_cache(signature: str, response: dict):
    if not _supabase:
        return
    try:
        _supabase.table("coaching_cache").upsert({"signature": signature, "response": response, "hit_count": 1}).execute()
    except Exception as e:
        print(f"Cache store error: {e}")


def check_daily_limit(user_ip: str) -> tuple[bool, int]:
    if not _supabase:
        return True, FREE_DAILY_LIMIT
    try:
        result = _supabase.table("daily_usage").select("call_count").eq("user_ip", user_ip).eq("usage_date", str(date.today())).execute()
        count = result.data[0]["call_count"] if result.data else 0
        return count < FREE_DAILY_LIMIT, FREE_DAILY_LIMIT - count
    except Exception:
        return True, FREE_DAILY_LIMIT


def increment_daily_usage(user_ip: str):
    if not _supabase:
        return
    try:
        _supabase.rpc("increment_usage", {"p_ip": user_ip, "p_date": str(date.today())}).execute()
    except Exception as e:
        print(f"Usage increment error: {e}")


def store_swing(user_ip: str, data: dict, coaching: dict):
    if not _supabase:
        return
    try:
        _supabase.table("swings").insert({
            "user_ip": user_ip,
            "score": data.get("score"),
            "grade": data.get("grade"),
            "handedness": data.get("handedness", "right"),
            "in_envelope_pct": data.get("in_envelope_pct"),
            "deviations": data.get("top_deviations", []),
            "claude_feedback": coaching,
        }).execute()
    except Exception as e:
        print(f"Swing store error: {e}")


def check_rate_limit(ip: str) -> bool:
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


@app.route("/manifest.json")
def serve_manifest():
    return send_from_directory(str(ROOT), "manifest.json")


@app.route("/reference_model.json")
def serve_model_combined():
    return _serve_model_file("reference_model.json")


@app.route("/reference_model_driver.json")
def serve_model_driver():
    return _serve_model_file("reference_model_driver.json")


@app.route("/reference_model_iron.json")
def serve_model_iron():
    return _serve_model_file("reference_model_iron.json")


def _serve_model_file(filename: str):
    path = ROOT / filename
    if not path.exists():
        return jsonify({"error": f"{filename} not found. Run build_reference_model.py first."}), 404
    return send_from_directory(str(ROOT), filename)


@app.route("/api/feedback", methods=["POST"])
def feedback():
    ip = request.headers.get("X-Forwarded-For", request.remote_addr) or "unknown"
    ip = ip.split(",")[0].strip()

    if not check_rate_limit(ip):
        return jsonify({"error": "Rate limit exceeded. Max 10 requests per hour."}), 429

    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key or api_key == "your_key_here":
        return jsonify({"error": "ANTHROPIC_API_KEY not set in .env"}), 500

    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "Empty request body"}), 400

    score        = data.get("score", 0)
    grade        = data.get("grade", "?")
    in_envelope  = data.get("in_envelope_pct", 0)
    deviations   = data.get("top_deviations", [])
    club_type    = data.get("club_type", "driver")
    club_label   = "driver" if club_type == "driver" else "iron"

    # Check daily limit
    under_limit, remaining = check_daily_limit(ip)
    if not under_limit:
        coaching = {
            "coaching": [],
            "summary": f"You've used your {FREE_DAILY_LIMIT} free analyses today. Come back tomorrow!",
            "limit_reached": True,
            "calls_remaining": 0,
            "from_cache": False,
        }
        store_swing(ip, data, coaching)
        return jsonify(coaching)

    # Check Supabase cache for this deviation pattern
    signature = make_signature(deviations)
    cached = check_supabase_cache(signature)
    if cached:
        cached["calls_remaining"] = remaining
        cached["from_cache"] = True
        cached.setdefault("limit_reached", False)
        store_swing(ip, data, cached)
        return jsonify(cached)

    # Cache miss — call Claude
    user_message = f"""Club: {club_label}
Swing score: {score}/100 (Grade: {grade})
Frames within pro envelope: {in_envelope:.1f}%

Top deviations from {club_label} professional benchmarks:
"""
    for i, dev in enumerate(deviations[:3], 1):
        user_message += (
            f"{i}. {dev.get('feature','?')}: {dev.get('direction','?')} "
            f"by avg {dev.get('mean_delta',0):.1f}° "
            f"(severity {dev.get('severity',0):.1f}) during {dev.get('phase','?')}\n"
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
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        coaching = json.loads(raw)
        coaching["limit_reached"] = False
        coaching["calls_remaining"] = remaining - 1
        coaching["from_cache"] = False

        increment_daily_usage(ip)
        store_supabase_cache(signature, coaching)
        store_swing(ip, data, coaching)

        return jsonify(coaching)

    except json.JSONDecodeError as e:
        return jsonify({"error": f"Claude returned invalid JSON: {e}"}), 502
    except anthropic.APIError as e:
        return jsonify({"error": f"Anthropic API error: {e}"}), 502
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/stats")
def stats():
    if not _supabase:
        return jsonify({"error": "Supabase not connected"})
    try:
        swing_count = _supabase.table("swings").select("id", count="exact").execute()
        cache_count = _supabase.table("coaching_cache").select("id", count="exact").execute()
        total_hits  = _supabase.table("coaching_cache").select("hit_count").execute()
        hits = sum(r["hit_count"] for r in total_hits.data) if total_hits.data else 0
        return jsonify({
            "total_swings": swing_count.count,
            "cached_patterns": cache_count.count,
            "total_cache_hits": hits,
        })
    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    print("\n=== Golf Swing Analyzer Server ===")
    print("  Serving on: http://localhost:8080")
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key or api_key == "your_key_here":
        print("  [warn] ANTHROPIC_API_KEY not set — coaching feedback will fail")
    else:
        print("  ✓ Anthropic client loaded")
    print(f"  {'✓ Supabase connected' if _supabase else '✗ Supabase not connected'}")
    for fname in ["reference_model.json", "reference_model_driver.json", "reference_model_iron.json"]:
        status = "✓" if (ROOT / fname).exists() else "✗ (not built yet)"
        print(f"  {fname}: {status}")
    print()
    port = int(os.getenv("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)
