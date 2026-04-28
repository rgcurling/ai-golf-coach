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

FREE_DAILY_LIMIT = 15
BYPASS_IPS = {ip.strip() for ip in os.getenv("BYPASS_IPS", "").split(",") if ip.strip()}

CLUB_CATEGORIES = ["driver", "fairway_wood", "long_iron", "mid_iron", "short_iron", "wedge"]

_supabase = None
try:
    _supabase = create_client(os.getenv("SUPABASE_URL", ""), os.getenv("SUPABASE_KEY", ""))
except Exception as _e:
    print(f"  [warn] Supabase init failed: {_e}")

CLAUDE_MODEL = "claude-sonnet-4-20250514"

SYSTEM_PROMPT = """You are a PGA-certified golf instructor analyzing a student's swing biomechanics data.
You will receive structured data showing how their swing deviates from professional benchmarks for a specific club.
When multiple swings are provided, the deviations are averaged across all swings — focus on consistent patterns rather than one-off errors.
Tailor your advice to the club being used — driver swings require more rotation and wider arc, iron swings require steeper attack angle and shaft lean, wedge swings require precision and controlled tempo.
Respond with exactly 3 coaching points, one per deviation, in this JSON format:
{
  "coaching": [
    { "priority": 1, "headline": "short 4-6 word issue title", "cue": "one specific actionable sentence the golfer can apply immediately" },
    { "priority": 2, "headline": "...", "cue": "..." },
    { "priority": 3, "headline": "...", "cue": "..." }
  ],
  "summary": "one encouraging sentence summarizing overall swing quality"
}"""

# ── Reference model loading ──────────────────────────────────────────────────

REFERENCE_MODELS: dict = {}

def _load_models():
    # New per-club models in reference_models/
    for club in CLUB_CATEGORIES:
        path = ROOT / "reference_models" / f"{club}.json"
        if path.exists():
            with open(path) as f:
                REFERENCE_MODELS[club] = json.load(f)
            print(f"  ✓ Loaded {club} model ({REFERENCE_MODELS[club]['n_swings']} swings)")

    # Legacy fallbacks
    if "driver" not in REFERENCE_MODELS:
        path = ROOT / "reference_model_driver.json"
        if path.exists():
            with open(path) as f:
                REFERENCE_MODELS["driver"] = json.load(f)
            print(f"  ✓ Loaded driver model (legacy)")

    legacy_iron = None
    iron_path = ROOT / "reference_model_iron.json"
    if iron_path.exists():
        with open(iron_path) as f:
            legacy_iron = json.load(f)

    iron_clubs = ["mid_iron", "long_iron", "short_iron", "fairway_wood", "wedge"]
    missing_iron = [c for c in iron_clubs if c not in REFERENCE_MODELS]
    if missing_iron and legacy_iron:
        for club in missing_iron:
            REFERENCE_MODELS[club] = legacy_iron
        print(f"  ✓ Applied legacy iron model to: {', '.join(missing_iron)}")

    # Combined fallback for any remaining gaps
    combined_path = ROOT / "reference_model.json"
    if combined_path.exists():
        with open(combined_path) as f:
            REFERENCE_MODELS["combined"] = json.load(f)


_load_models()


def get_reference_model(club_category: str) -> dict | None:
    if club_category in REFERENCE_MODELS:
        return REFERENCE_MODELS[club_category]
    # Fallback chain for clubs without a dedicated model
    fallbacks = {
        "fairway_wood": "driver",
        "long_iron": "mid_iron",
        "short_iron": "mid_iron",
    }
    fb = fallbacks.get(club_category)
    if fb and fb in REFERENCE_MODELS:
        return REFERENCE_MODELS[fb]
    return REFERENCE_MODELS.get("combined") or (next(iter(REFERENCE_MODELS.values()), None))


# ── Tempo feedback ───────────────────────────────────────────────────────────

def get_tempo_feedback(tempo_ratio: float, club_model: dict) -> dict | None:
    tempo = club_model.get("tempo", {})
    p25 = tempo.get("ratio_p25", 2.5)
    p75 = tempo.get("ratio_p75", 3.5)
    p50 = tempo.get("ratio_p50", 3.0)

    if tempo_ratio < p25:
        return {
            "feature": "tempo",
            "direction": "low",
            "headline": "Rushing the downswing",
            "cue": (f"Your tempo ratio is {tempo_ratio:.1f}:1. "
                    f"Tour average is {p50:.1f}:1. "
                    "Pause slightly at the top before starting down."),
            "severity": round((p25 - tempo_ratio) / p25 * 10, 1),
        }
    if tempo_ratio > p75:
        return {
            "feature": "tempo",
            "direction": "high",
            "headline": "Swing too slow",
            "cue": (f"Your tempo ratio is {tempo_ratio:.1f}:1. "
                    f"Tour average is {p50:.1f}:1. "
                    "Let the downswing accelerate more naturally."),
            "severity": round((tempo_ratio - p75) / p75 * 10, 1),
        }
    return None


# ── Supabase helpers ─────────────────────────────────────────────────────────

def make_signature(deviations: list, club_category: str) -> str:
    sig = json.dumps({
        "club": club_category,
        "devs": [
            {"feature": d["feature"], "direction": d["direction"],
             "severity_bucket": round(d.get("severity", 0))}
            for d in sorted(deviations[:3], key=lambda x: x["feature"])
        ],
    }, sort_keys=True)
    return hashlib.md5(sig.encode()).hexdigest()


def check_supabase_cache(signature: str):
    if not _supabase:
        return None
    try:
        result = _supabase.table("coaching_cache").select("response, hit_count").eq("signature", signature).execute()
        if result.data:
            _supabase.table("coaching_cache").update(
                {"hit_count": result.data[0]["hit_count"] + 1}
            ).eq("signature", signature).execute()
            return result.data[0]["response"]
    except Exception as e:
        print(f"Cache check error: {e}")
    return None


def store_supabase_cache(signature: str, response: dict):
    if not _supabase:
        return
    try:
        _supabase.table("coaching_cache").upsert(
            {"signature": signature, "response": response, "hit_count": 1}
        ).execute()
    except Exception as e:
        print(f"Cache store error: {e}")


def check_daily_limit(user_ip: str) -> tuple[bool, int]:
    if user_ip in BYPASS_IPS:
        return True, 999
    if not _supabase:
        return True, FREE_DAILY_LIMIT
    try:
        result = _supabase.table("daily_usage").select("call_count").eq("user_ip", user_ip).eq(
            "usage_date", str(date.today())).execute()
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
            "club_category": data.get("club_category"),
            "tempo_ratio": data.get("tempo_ratio"),
            "deviations": data.get("top_deviations", []),
            "claude_feedback": coaching,
        }).execute()
    except Exception as e:
        print(f"Swing store error: {e}")


def check_rate_limit(ip: str) -> bool:
    if ip in BYPASS_IPS:
        return True
    now = time.time()
    window_start = now - RATE_WINDOW
    _rate_buckets[ip] = [t for t in _rate_buckets[ip] if t > window_start]
    if len(_rate_buckets[ip]) >= RATE_LIMIT:
        return False
    _rate_buckets[ip].append(now)
    return True


# ── Static file routes ───────────────────────────────────────────────────────

@app.route("/")
def serve_index():
    return send_from_directory(str(ROOT), "index.html")


@app.route("/manifest.json")
def serve_manifest():
    return send_from_directory(str(ROOT), "manifest.json")


@app.route("/reference_model.json")
def serve_model_combined():
    return _serve_model_file(ROOT, "reference_model.json")


@app.route("/reference_model_driver.json")
def serve_model_driver():
    return _serve_model_file(ROOT, "reference_model_driver.json")


@app.route("/reference_model_iron.json")
def serve_model_iron():
    return _serve_model_file(ROOT, "reference_model_iron.json")


@app.route("/reference_models/<club>.json")
def serve_club_model(club):
    if club not in CLUB_CATEGORIES:
        return jsonify({"error": f"Unknown club category: {club}"}), 404
    return _serve_model_file(ROOT / "reference_models", f"{club}.json")


def _serve_model_file(directory: Path, filename: str):
    path = Path(directory) / filename
    if not path.exists():
        return jsonify({"error": f"{filename} not found"}), 404
    return send_from_directory(str(directory), filename)


# ── API endpoints ────────────────────────────────────────────────────────────

@app.route("/api/models")
def models():
    available = [club for club in CLUB_CATEGORIES if club in REFERENCE_MODELS]
    return jsonify({"available_clubs": available})


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
    tempo_ratio  = data.get("tempo_ratio")
    swing_count  = int(data.get("swing_count", 1))
    per_swing_scores = data.get("per_swing_scores", [])

    # Accept both new club_category and legacy club_type
    club_category = data.get("club_category") or data.get("club_type", "driver")
    if club_category == "iron":
        club_category = "mid_iron"
    club_label = club_category.replace("_", " ")

    club_model = get_reference_model(club_category)

    # Daily limit check
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

    # Cache check
    signature = make_signature(deviations, club_category)
    cached = check_supabase_cache(signature)
    if cached:
        # Add tempo coaching to cached response if needed
        if tempo_ratio is not None and club_model:
            tempo_fb = get_tempo_feedback(float(tempo_ratio), club_model)
            if tempo_fb:
                cached.setdefault("coaching", [])
                cached["coaching"].append({
                    "priority": len(cached["coaching"]) + 1,
                    "headline": tempo_fb["headline"],
                    "cue": tempo_fb["cue"],
                })
        cached["calls_remaining"] = remaining
        cached["from_cache"] = True
        cached.setdefault("limit_reached", False)
        store_swing(ip, data, cached)
        return jsonify(cached)

    # Build Claude prompt
    swing_context = (
        f"Swings analyzed: {swing_count} (individual scores: {', '.join(str(s) for s in per_swing_scores)})\n"
        if swing_count > 1 else ""
    )
    user_message = f"""Club: {club_label}
{swing_context}Average swing score: {score}/100 (Grade: {grade})
Frames within pro envelope: {in_envelope:.1f}%

Top deviations from {club_label} professional benchmarks{' (averaged across all swings)' if swing_count > 1 else ''}:
"""
    for i, dev in enumerate(deviations[:3], 1):
        user_message += (
            f"{i}. {dev.get('feature', '?')}: {dev.get('direction', '?')} "
            f"by avg {dev.get('mean_delta', 0):.1f}° "
            f"(severity {dev.get('severity', 0):.1f}) during {dev.get('phase', '?')}\n"
        )

    try:
        client_obj = anthropic.Anthropic(api_key=api_key)
        message = client_obj.messages.create(
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

        # Add tempo coaching if the user's tempo is outside the pro range
        if tempo_ratio is not None and club_model:
            tempo_fb = get_tempo_feedback(float(tempo_ratio), club_model)
            if tempo_fb:
                coaching["coaching"].append({
                    "priority": len(coaching["coaching"]) + 1,
                    "headline": tempo_fb["headline"],
                    "cue": tempo_fb["cue"],
                })

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
        print("  ✓ Anthropic client ready")
    print(f"  {'✓ Supabase connected' if _supabase else '✗ Supabase not connected'}")
    loaded = [c for c in CLUB_CATEGORIES if c in REFERENCE_MODELS]
    if loaded:
        print(f"  ✓ Models loaded: {', '.join(loaded)}")
    else:
        print("  [warn] No reference models found — run build_reference_model.py")
    print()
    port = int(os.getenv("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)
