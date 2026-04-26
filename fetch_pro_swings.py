"""
Downloads publicly available golf swing videos from YouTube for building the reference model.
Run after setup.py: python fetch_pro_swings.py

Targets slow-motion professional swings. Filters for 10-60 second clips likely to show
a single golfer with the full body visible.
"""
import json
import subprocess
import sys
import re
from pathlib import Path
from datetime import datetime

OUTPUT_DIR = Path("pro_swings")
MANIFEST_FILE = OUTPUT_DIR / "manifest.json"
MAX_VIDEOS = 25

# Search queries targeting high-quality, analyzable pro swings
SEARCH_QUERIES = [
    "PGA tour slow motion swing down the line",
    "professional golf swing face on slow motion",
    "tour pro golf swing analysis side view",
    "golf swing slow motion 4k down the line",
    "PGA pro swing slow motion face on",
]

# yt-dlp format: best video up to 720p, no audio needed
YTDLP_FORMAT = "bestvideo[height<=720][ext=mp4]/bestvideo[height<=720]/best[height<=720]"


def detect_camera_angle(title: str) -> str:
    title_lower = title.lower()
    if any(k in title_lower for k in ["down the line", "dtl", "side view", "behind"]):
        return "down_the_line"
    if any(k in title_lower for k in ["face on", "front view", "face-on"]):
        return "face_on"
    return "unknown"


def guess_player_name(title: str) -> str:
    """Heuristic: look for common golfer surnames in the video title."""
    pros = [
        "McIlroy", "Scheffler", "Rahm", "Koepka", "Thomas", "Spieth",
        "DeChambeau", "Morikawa", "Hovland", "Burns", "Fleetwood",
        "Woods", "Nicklaus", "Player", "Palmer", "Watson", "Els",
        "Mickelson", "Rose", "Westwood", "Stenson",
    ]
    for name in pros:
        if name.lower() in title.lower():
            return name
    return "Unknown"


def search_and_download(query: str, existing_ids: set) -> list[dict]:
    """Run yt-dlp in search mode and download matching videos. Returns list of metadata dicts."""
    print(f"\n  Searching: \"{query}\"")

    # First, list matching videos without downloading to filter by duration
    list_cmd = [
        "yt-dlp",
        f"ytsearch10:{query}",
        "--no-download",
        "--print", "%(id)s\t%(title)s\t%(duration)s\t%(uploader)s",
        "--no-warnings",
        "--quiet",
    ]
    try:
        result = subprocess.run(list_cmd, capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        print("  [warn] Search timed out, skipping query")
        return []
    except FileNotFoundError:
        print("  [error] yt-dlp not found. Make sure the venv is activated.")
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

        # Only keep videos between 10 and 60 seconds — longer vids are compilations
        if not (10 <= duration <= 60):
            continue

        candidates.append({
            "id": vid_id,
            "title": title,
            "duration": duration,
            "uploader": uploader,
        })

    print(f"  Found {len(candidates)} candidate(s) in 10-60s range")

    downloaded = []
    for c in candidates:
        if len(downloaded) >= 5:  # cap per query to spread across queries
            break

        vid_id = c["id"]
        url = f"https://www.youtube.com/watch?v={vid_id}"
        out_path = OUTPUT_DIR / f"{vid_id}.mp4"

        if out_path.exists():
            print(f"  [skip] {vid_id} already exists")
            existing_ids.add(vid_id)
            downloaded.append({
                "video_id": vid_id,
                "file": str(out_path),
                "title": c["title"],
                "player": guess_player_name(c["title"]),
                "camera_angle": detect_camera_angle(c["title"]),
                "duration_s": c["duration"],
                "uploader": c["uploader"],
                "source_url": url,
                "downloaded_at": datetime.utcnow().isoformat(),
            })
            continue

        print(f"  Downloading {vid_id}: {c['title'][:60]}")
        dl_cmd = [
            "yt-dlp",
            url,
            "-f", YTDLP_FORMAT,
            "-o", str(out_path),
            "--no-warnings",
            "--quiet",
        ]
        try:
            subprocess.run(dl_cmd, timeout=120, check=True)
        except subprocess.CalledProcessError:
            print(f"  [warn] Failed to download {vid_id}, skipping")
            continue
        except subprocess.TimeoutExpired:
            print(f"  [warn] Download timed out for {vid_id}, skipping")
            continue

        if not out_path.exists():
            continue

        existing_ids.add(vid_id)
        meta = {
            "video_id": vid_id,
            "file": str(out_path),
            "title": c["title"],
            "player": guess_player_name(c["title"]),
            "camera_angle": detect_camera_angle(c["title"]),
            "duration_s": c["duration"],
            "uploader": c["uploader"],
            "source_url": url,
            "downloaded_at": datetime.utcnow().isoformat(),
        }
        downloaded.append(meta)
        print(f"  [ok] {vid_id} saved ({c['duration']:.1f}s, {detect_camera_angle(c['title'])})")

    return downloaded


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Load existing manifest to avoid re-downloading
    manifest = []
    existing_ids: set = set()
    if MANIFEST_FILE.exists():
        try:
            manifest = json.loads(MANIFEST_FILE.read_text())
            existing_ids = {m["video_id"] for m in manifest}
            print(f"Resuming — {len(manifest)} videos already in manifest")
        except json.JSONDecodeError:
            pass

    print(f"\nTargeting {MAX_VIDEOS} videos across {len(SEARCH_QUERIES)} search queries\n")

    for query in SEARCH_QUERIES:
        if len(manifest) >= MAX_VIDEOS:
            print(f"\nReached target of {MAX_VIDEOS} videos, stopping early.")
            break
        results = search_and_download(query, existing_ids)
        manifest.extend(results)
        MANIFEST_FILE.write_text(json.dumps(manifest, indent=2))

    # Summary
    angles = {}
    for m in manifest:
        a = m.get("camera_angle", "unknown")
        angles[a] = angles.get(a, 0) + 1

    print(f"\n=== Download complete ===")
    print(f"  Total videos: {len(manifest)}")
    for angle, count in angles.items():
        print(f"  {angle}: {count}")
    print(f"\nManifest saved to {MANIFEST_FILE}")
    print("Next: python build_reference_model.py\n")


if __name__ == "__main__":
    main()
