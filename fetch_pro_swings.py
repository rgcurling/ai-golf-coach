"""
Downloads publicly available golf swing videos from YouTube for building per-club reference models.

Usage:
  python fetch_pro_swings.py --club driver
  python fetch_pro_swings.py --club mid_iron
  python fetch_pro_swings.py --club all
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime, timezone

BASE_DIR = Path("pro_swings")

CLUB_CONFIGS = {
    "driver": {
        "queries": [
            "driver swing slow motion down the line tour pro",
            "PGA tour driver swing face on slow motion",
            "Rory McIlroy driver swing slow motion",
            "Dustin Johnson driver swing analysis",
            "professional driver swing sequence",
        ],
        "target_videos": 40,
    },
    "fairway_wood": {
        "queries": [
            "3 wood swing slow motion professional",
            "fairway wood swing PGA tour face on",
            "tour pro 3 wood swing analysis",
        ],
        "target_videos": 25,
    },
    "long_iron": {
        "queries": [
            "long iron swing slow motion professional",
            "4 iron swing PGA tour down the line",
            "tour pro iron swing slow motion 4 iron",
        ],
        "target_videos": 25,
    },
    "mid_iron": {
        "queries": [
            "7 iron swing slow motion PGA tour",
            "professional golf swing 7 iron face on",
            "Adam Scott iron swing slow motion",
            "Collin Morikawa iron swing analysis",
            "tour pro 7 iron down the line",
        ],
        "target_videos": 40,
    },
    "short_iron": {
        "queries": [
            "9 iron swing slow motion professional",
            "short iron swing PGA tour analysis",
            "tour pro 9 iron swing face on",
        ],
        "target_videos": 25,
    },
    "wedge": {
        "queries": [
            "pitching wedge swing slow motion tour pro",
            "wedge swing professional golf face on",
            "PGA tour wedge swing analysis slow motion",
            "professional golf wedge technique slow motion",
        ],
        "target_videos": 30,
    },
}

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


def search_and_download(query: str, club: str, output_dir: Path, existing_ids: set) -> list[dict]:
    print(f"\n  [{club.upper()}] Searching: \"{query}\"")

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
        if len(downloaded) >= 3:
            break

        vid_id = c["id"]
        url = f"https://www.youtube.com/watch?v={vid_id}"
        out_path = output_dir / f"{vid_id}.mp4"

        if out_path.exists():
            print(f"  [skip] {vid_id} already downloaded")
            existing_ids.add(vid_id)
            downloaded.append(_meta(vid_id, c, url, out_path, club))
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
        meta = _meta(vid_id, c, url, out_path, club)
        downloaded.append(meta)
        print(f"  [ok] {vid_id} ({c['duration']:.1f}s, {detect_camera_angle(c['title'])}, {club})")

    return downloaded


def _meta(vid_id, c, url, out_path, club) -> dict:
    return {
        "video_id": vid_id,
        "file": str(out_path),
        "title": c["title"],
        "player": guess_player_name(c["title"]),
        "camera_angle": detect_camera_angle(c["title"]),
        "club_category": club,
        "duration_s": c["duration"],
        "uploader": c["uploader"],
        "source_url": url,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
    }


def run_club(club: str) -> int:
    config = CLUB_CONFIGS[club]
    output_dir = BASE_DIR / club
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_file = output_dir / "manifest.json"

    manifest = []
    existing_ids: set = set()
    if manifest_file.exists():
        try:
            manifest = json.loads(manifest_file.read_text())
            existing_ids = {m["video_id"] for m in manifest}
            print(f"  Resuming — {len(manifest)} videos already in manifest")
        except json.JSONDecodeError:
            pass

    target = config["target_videos"]
    print(f"\n=== {club.upper().replace('_', ' ')} ===")
    print(f"  Target: {target}  |  Have: {len(manifest)}")

    for query in config["queries"]:
        if len(manifest) >= target:
            print(f"  Reached target of {target} videos, skipping remaining queries")
            break
        results = search_and_download(query, club, output_dir, existing_ids)
        manifest.extend(results)
        manifest_file.write_text(json.dumps(manifest, indent=2))

    print(f"\n  {club}: {len(manifest)} videos downloaded")
    return len(manifest)


def main():
    parser = argparse.ArgumentParser(description="Download pro golf swing videos per club category")
    parser.add_argument(
        "--club", required=True,
        choices=list(CLUB_CONFIGS.keys()) + ["all"],
        help="Club category to download, or 'all'",
    )
    args = parser.parse_args()

    BASE_DIR.mkdir(exist_ok=True)
    clubs = list(CLUB_CONFIGS.keys()) if args.club == "all" else [args.club]

    totals = {}
    for club in clubs:
        totals[club] = run_club(club)

    print("\n=== Download Summary ===")
    for club, count in totals.items():
        print(f"  {club:<15} {count:>3} videos  →  pro_swings/{club}/")
    print(f"\nNext: python build_reference_model.py --club {args.club}\n")


if __name__ == "__main__":
    main()
