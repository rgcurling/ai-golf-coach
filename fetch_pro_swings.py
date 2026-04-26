"""
Downloads publicly available golf swing videos from YouTube for building the reference model.
Run after setup.py: python fetch_pro_swings.py

Queries are split by club type so build_reference_model.py can produce
separate driver and iron envelopes.
"""
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime, timezone

OUTPUT_DIR = Path("pro_swings")
MANIFEST_FILE = OUTPUT_DIR / "manifest.json"
MAX_VIDEOS = 60  # 30 driver + 30 iron

# Each entry is (search_query, club_type).
# Queries target specific players for biomechanical variety.
SEARCH_QUERIES = [
    # ── DRIVER ────────────────────────────────────────────────
    ("Rory McIlroy driver slow motion down the line",           "driver"),
    ("Rory McIlroy driver swing face on slow motion",           "driver"),
    ("Jon Rahm driver slow motion golf swing down the line",    "driver"),
    ("Jon Rahm driver swing face on slow motion",               "driver"),
    ("Scottie Scheffler driver slow motion swing down the line","driver"),
    ("Scottie Scheffler driver golf swing face on slow motion", "driver"),
    ("Justin Thomas driver slow motion golf swing",             "driver"),
    ("Xander Schauffele driver slow motion golf swing",         "driver"),
    ("Collin Morikawa driver slow motion golf swing",           "driver"),
    ("Adam Scott driver slow motion golf swing down the line",  "driver"),
    ("Fred Couples driver slow motion golf swing",              "driver"),
    ("Ernie Els driver slow motion golf swing",                 "driver"),
    ("Nelly Korda driver slow motion golf swing",               "driver"),
    ("Brooks Koepka driver slow motion golf swing",             "driver"),
    ("Dustin Johnson driver slow motion swing",                 "driver"),

    # ── IRON ──────────────────────────────────────────────────
    ("Collin Morikawa iron swing slow motion down the line",    "iron"),
    ("Collin Morikawa iron swing face on slow motion",          "iron"),
    ("Justin Thomas iron swing slow motion down the line",      "iron"),
    ("Justin Thomas iron swing face on slow motion",            "iron"),
    ("Rory McIlroy iron swing slow motion",                     "iron"),
    ("Jon Rahm iron swing slow motion down the line",           "iron"),
    ("Jon Rahm iron swing face on slow motion",                 "iron"),
    ("Scottie Scheffler iron swing slow motion",                "iron"),
    ("Adam Scott iron swing slow motion down the line",         "iron"),
    ("Xander Schauffele iron swing slow motion",                "iron"),
    ("Tiger Woods iron swing slow motion",                      "iron"),
    ("Jordan Spieth iron swing slow motion",                    "iron"),
    ("Nelly Korda iron swing slow motion",                      "iron"),
    ("Viktor Hovland iron swing slow motion",                   "iron"),
    ("Tommy Fleetwood iron swing slow motion",                  "iron"),
]

YTDLP_FORMAT = "bestvideo[height<=720][ext=mp4]/bestvideo[height<=720]/best[height<=720]"


def detect_camera_angle(title: str) -> str:
    t = title.lower()
    if any(k in t for k in ["down the line", "dtl", "side view", "behind"]):
        return "down_the_line"
    if any(k in t for k in ["face on", "front view", "face-on"]):
        return "face_on"
    return "unknown"


def guess_player_name(title: str) -> str:
    pros = [
        "McIlroy", "Scheffler", "Rahm", "Koepka", "Thomas", "Spieth",
        "DeChambeau", "Morikawa", "Hovland", "Burns", "Fleetwood",
        "Woods", "Nicklaus", "Player", "Palmer", "Watson", "Els",
        "Mickelson", "Rose", "Westwood", "Stenson", "Johnson", "Schauffele",
        "Scott", "Couples", "Korda",
    ]
    for name in pros:
        if name.lower() in title.lower():
            return name
    return "Unknown"


def search_and_download(query: str, club_type: str, existing_ids: set) -> list[dict]:
    print(f"\n  [{club_type.upper()}] Searching: \"{query}\"")

    list_cmd = [
        "yt-dlp", f"ytsearch10:{query}",
        "--no-download",
        "--print", "%(id)s\t%(title)s\t%(duration)s\t%(uploader)s",
        "--no-warnings", "--quiet",
    ]
    try:
        result = subprocess.run(list_cmd, capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        print("  [warn] Search timed out, skipping")
        return []
    except FileNotFoundError:
        print("  [error] yt-dlp not found. Activate the venv first.")
        sys.exit(1)

    candidates = []
    for line in result.stdout.strip().splitlines():
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        vid_id, title, duration_str, uploader = parts[0], parts[1], parts[2], parts[3]
        if vid_id in existing_ids:
            continue
        try:
            duration = float(duration_str)
        except ValueError:
            continue
        if not (8 <= duration <= 90):
            continue
        candidates.append({"id": vid_id, "title": title, "duration": duration, "uploader": uploader})

    print(f"  Found {len(candidates)} candidate(s) in 8-90s range")

    downloaded = []
    for c in candidates:
        if len(downloaded) >= 3:  # max 3 per query so club types stay balanced
            break

        vid_id = c["id"]
        url = f"https://www.youtube.com/watch?v={vid_id}"
        out_path = OUTPUT_DIR / f"{vid_id}.mp4"

        if out_path.exists():
            print(f"  [skip] {vid_id} already downloaded")
            existing_ids.add(vid_id)
            downloaded.append(_meta(vid_id, c, url, out_path, club_type))
            continue

        print(f"  Downloading {vid_id}: {c['title'][:60]}")
        dl_cmd = ["yt-dlp", url, "-f", YTDLP_FORMAT, "-o", str(out_path), "--no-warnings", "--quiet"]
        try:
            subprocess.run(dl_cmd, timeout=120, check=True)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            print(f"  [warn] Download failed: {e}")
            continue

        if not out_path.exists():
            continue

        existing_ids.add(vid_id)
        meta = _meta(vid_id, c, url, out_path, club_type)
        downloaded.append(meta)
        print(f"  [ok] {vid_id} ({c['duration']:.1f}s, {detect_camera_angle(c['title'])}, {club_type})")

    return downloaded


def _meta(vid_id, c, url, out_path, club_type) -> dict:
    return {
        "video_id": vid_id,
        "file": str(out_path),
        "title": c["title"],
        "player": guess_player_name(c["title"]),
        "camera_angle": detect_camera_angle(c["title"]),
        "club_type": club_type,
        "duration_s": c["duration"],
        "uploader": c["uploader"],
        "source_url": url,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
    }


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    manifest = []
    existing_ids: set = set()
    if MANIFEST_FILE.exists():
        try:
            manifest = json.loads(MANIFEST_FILE.read_text())
            existing_ids = {m["video_id"] for m in manifest}
            print(f"Resuming — {len(manifest)} videos already in manifest")
        except json.JSONDecodeError:
            pass

    print(f"\nTargeting {MAX_VIDEOS} videos across {len(SEARCH_QUERIES)} queries\n")

    driver_count = sum(1 for m in manifest if m.get("club_type") == "driver")
    iron_count   = sum(1 for m in manifest if m.get("club_type") == "iron")

    for query, club_type in SEARCH_QUERIES:
        # Skip if we already have enough of this club type
        current = driver_count if club_type == "driver" else iron_count
        if current >= MAX_VIDEOS // 2:
            continue

        results = search_and_download(query, club_type, existing_ids)
        manifest.extend(results)
        for r in results:
            if r["club_type"] == "driver": driver_count += 1
            else: iron_count += 1
        MANIFEST_FILE.write_text(json.dumps(manifest, indent=2))

    print(f"\n=== Download complete ===")
    print(f"  Total videos: {len(manifest)}")
    print(f"  Driver: {driver_count}  |  Iron: {iron_count}")
    print(f"\nManifest saved to {MANIFEST_FILE}")
    print("Next: python build_reference_model.py\n")


if __name__ == "__main__":
    main()
