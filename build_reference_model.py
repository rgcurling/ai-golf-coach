"""
Processes downloaded pro swing videos and builds per-club reference envelopes.

Usage:
  python build_reference_model.py --club driver
  python build_reference_model.py --club mid_iron
  python build_reference_model.py --club all
"""
import argparse
import json
import math
import urllib.request
import cv2
import numpy as np
from pathlib import Path
from datetime import date
from tqdm import tqdm

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

MODEL_PATH = Path("pose_landmarker_heavy.task")
MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
MODELS_DIR = Path("reference_models")

N_FRAMES         = 100
IMPACT_FRAME     = 75
MIN_VALID_FRAMES = 20

CLUB_CATEGORIES = ["driver", "fairway_wood", "long_iron", "mid_iron", "short_iron", "wedge"]

# tempo_ratio is the 12th feature: backswing_frames / downswing_frames (stored as constant per swing)
FEATURES_BASE = [
    "lead_elbow", "trail_elbow",
    "lead_wrist_angle", "trail_wrist_angle",
    "lead_knee_flex", "trail_knee_flex",
    "spine_tilt", "hip_rotation", "shoulder_rotation",
    "x_factor", "hand_height",
]
FEATURES = FEATURES_BASE + ["tempo_ratio"]

LM = {
    "LEFT_SHOULDER": 11,  "RIGHT_SHOULDER": 12,
    "LEFT_ELBOW":    13,  "RIGHT_ELBOW":    14,
    "LEFT_WRIST":    15,  "RIGHT_WRIST":    16,
    "LEFT_HIP":      23,  "RIGHT_HIP":      24,
    "LEFT_KNEE":     25,  "RIGHT_KNEE":     26,
    "LEFT_ANKLE":    27,  "RIGHT_ANKLE":    28,
}

# Per-club scoring weights (higher = more important for that club)
_WEIGHTS_DRIVER = {
    "x_factor": 2.0,
    "hip_rotation": 1.8,
    "shoulder_rotation": 1.6,
    "spine_tilt": 1.4,
    "hand_height": 1.3,
    "lead_knee_flex": 1.3,
    "lead_elbow": 1.2,
    "trail_elbow": 1.1,
    "trail_knee_flex": 1.1,
    "lead_wrist_angle": 1.0,
    "trail_wrist_angle": 1.0,
    "tempo_ratio": 1.5,
}
_WEIGHTS_MID_IRON = {
    "spine_tilt": 1.8,
    "x_factor": 1.6,
    "lead_elbow": 1.5,
    "trail_elbow": 1.3,
    "hip_rotation": 1.4,
    "shoulder_rotation": 1.3,
    "trail_wrist_angle": 1.3,
    "lead_wrist_angle": 1.2,
    "hand_height": 1.2,
    "lead_knee_flex": 1.2,
    "trail_knee_flex": 1.0,
    "tempo_ratio": 1.4,
}
_WEIGHTS_WEDGE = {
    "spine_tilt": 2.0,
    "trail_wrist_angle": 1.8,
    "lead_elbow": 1.6,
    "tempo_ratio": 1.6,
    "hand_height": 1.5,
    "lead_wrist_angle": 1.4,
    "hip_rotation": 1.3,
    "x_factor": 1.2,
    "shoulder_rotation": 1.2,
    "lead_knee_flex": 1.1,
    "trail_knee_flex": 1.0,
    "trail_elbow": 1.0,
}


def _interpolate_weights(w1: dict, w2: dict, t: float) -> dict:
    keys = set(w1) | set(w2)
    return {k: round(w1.get(k, 1.0) * (1 - t) + w2.get(k, 1.0) * t, 3) for k in keys}


def get_club_weights(club: str) -> dict:
    if club == "driver":
        return _WEIGHTS_DRIVER
    if club == "fairway_wood":
        w = dict(_WEIGHTS_DRIVER)
        w["x_factor"] = 1.7
        return w
    if club == "long_iron":
        return _interpolate_weights(_WEIGHTS_DRIVER, _WEIGHTS_MID_IRON, 0.5)
    if club == "mid_iron":
        return _WEIGHTS_MID_IRON
    if club == "short_iron":
        return _interpolate_weights(_WEIGHTS_MID_IRON, _WEIGHTS_WEDGE, 0.5)
    if club == "wedge":
        return _WEIGHTS_WEDGE
    return {f: 1.0 for f in FEATURES}


def ensure_model():
    if not MODEL_PATH.exists():
        print("Downloading pose landmarker model (~30MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print(f"  Saved to {MODEL_PATH}")


def angle_3pts(a, b, c) -> float:
    ba = np.array([a.x - b.x, a.y - b.y, a.z - b.z])
    bc = np.array([c.x - b.x, c.y - b.y, c.z - b.z])
    nb, nc = np.linalg.norm(ba), np.linalg.norm(bc)
    if nb < 1e-6 or nc < 1e-6:
        return float("nan")
    return math.degrees(math.acos(np.clip(np.dot(ba, bc) / (nb * nc), -1.0, 1.0)))


def is_left_handed(norm_lms) -> bool:
    return norm_lms[LM["LEFT_SHOULDER"]].x < norm_lms[LM["RIGHT_SHOULDER"]].x


def extract_angles(world_lms, norm_lms) -> dict:
    left_handed = is_left_handed(norm_lms)
    w = world_lms

    if left_handed:
        ls, rs = LM["RIGHT_SHOULDER"], LM["LEFT_SHOULDER"]
        le, re = LM["RIGHT_ELBOW"],    LM["LEFT_ELBOW"]
        lw, rw = LM["RIGHT_WRIST"],    LM["LEFT_WRIST"]
        lh, rh = LM["RIGHT_HIP"],      LM["LEFT_HIP"]
        lk, rk = LM["RIGHT_KNEE"],     LM["LEFT_KNEE"]
        la, ra = LM["RIGHT_ANKLE"],    LM["LEFT_ANKLE"]
    else:
        ls, rs = LM["LEFT_SHOULDER"],  LM["RIGHT_SHOULDER"]
        le, re = LM["LEFT_ELBOW"],     LM["RIGHT_ELBOW"]
        lw, rw = LM["LEFT_WRIST"],     LM["RIGHT_WRIST"]
        lh, rh = LM["LEFT_HIP"],       LM["RIGHT_HIP"]
        lk, rk = LM["LEFT_KNEE"],      LM["RIGHT_KNEE"]
        la, ra = LM["LEFT_ANKLE"],     LM["RIGHT_ANKLE"]

    lead_elbow        = angle_3pts(w[ls], w[le], w[lw])
    trail_elbow       = angle_3pts(w[rs], w[re], w[rw])
    lead_wrist_angle  = angle_3pts(w[le], w[lw], w[ls])
    trail_wrist_angle = angle_3pts(w[re], w[rw], w[rs])
    lead_knee_flex    = angle_3pts(w[lh], w[lk], w[la])
    trail_knee_flex   = angle_3pts(w[rh], w[rk], w[ra])

    mid_sx = (w[ls].x + w[rs].x) / 2
    mid_sy = (w[ls].y + w[rs].y) / 2
    mid_sz = (w[ls].z + w[rs].z) / 2
    mid_hx = (w[lh].x + w[rh].x) / 2
    mid_hy = (w[lh].y + w[rh].y) / 2
    mid_hz = (w[lh].z + w[rh].z) / 2
    sv = np.array([mid_hx - mid_sx, mid_hy - mid_sy, mid_hz - mid_sz])
    sv_len = np.linalg.norm(sv)
    spine_tilt = float("nan") if sv_len < 1e-6 else math.degrees(math.acos(np.clip(sv[1] / sv_len, -1, 1)))

    hip_rotation      = math.degrees(math.atan2(w[lh].z - w[rh].z, abs(w[lh].x - w[rh].x) + 1e-6))
    shoulder_rotation = math.degrees(math.atan2(w[ls].z - w[rs].z, abs(w[ls].x - w[rs].x) + 1e-6))
    x_factor          = shoulder_rotation - hip_rotation

    avg_wy     = (w[lw].y + w[rw].y) / 2
    shoulder_y = (w[ls].y + w[rs].y) / 2
    hip_y      = (w[lh].y + w[rh].y) / 2
    y_range    = abs(hip_y - shoulder_y)
    hand_height = float("nan") if y_range < 1e-6 else (shoulder_y - avg_wy) / y_range

    return dict(
        lead_elbow=lead_elbow, trail_elbow=trail_elbow,
        lead_wrist_angle=lead_wrist_angle, trail_wrist_angle=trail_wrist_angle,
        lead_knee_flex=lead_knee_flex, trail_knee_flex=trail_knee_flex,
        spine_tilt=spine_tilt,
        hip_rotation=hip_rotation, shoulder_rotation=shoulder_rotation,
        x_factor=x_factor, hand_height=hand_height,
    )


def detect_impact_frame(frames: list[dict]) -> int:
    n = len(frames)
    if n < 4:
        return n // 2
    vals = [f.get("lead_elbow", float("nan")) for f in frames]
    velocities = [(i, abs(vals[i + 1] - vals[i - 1]))
                  for i in range(1, n - 1)
                  if not (math.isnan(vals[i - 1]) or math.isnan(vals[i + 1]))]
    downswing = [(i, v) for i, v in velocities if i >= int(n * 0.4)]
    return max(downswing, key=lambda x: x[1])[0] if downswing else int(n * 0.75)


def normalize_swing(frames: list[dict], impact_idx: int, tempo_ratio: float) -> np.ndarray:
    normalized = np.full((N_FRAMES, len(FEATURES)), float("nan"))

    for fi, feat in enumerate(FEATURES_BASE):
        arr = np.array([f.get(feat, float("nan")) for f in frames])

        pre = arr[:impact_idx + 1]
        vx = np.where(~np.isnan(pre))[0]
        if len(vx) >= 2:
            normalized[:IMPACT_FRAME + 1, fi] = np.interp(
                np.linspace(0, impact_idx, IMPACT_FRAME + 1), vx, pre[vx])

        post = arr[impact_idx:]
        vx = np.where(~np.isnan(post))[0]
        if len(vx) >= 2:
            normalized[IMPACT_FRAME:, fi] = np.interp(
                np.linspace(0, len(post) - 1, N_FRAMES - IMPACT_FRAME), vx, post[vx])

    # Fill tempo_ratio (12th feature) as constant across all frames
    normalized[:, len(FEATURES_BASE)] = tempo_ratio
    return normalized


def process_video(video_path: str) -> tuple[np.ndarray, float] | None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    options = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    raw_frames, frame_idx = [], 0
    with mp_vision.PoseLandmarker.create_from_options(options) as lm:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = lm.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
                int(frame_idx * 1000 / fps),
            )
            frame_idx += 1
            if result.pose_world_landmarks and result.pose_landmarks:
                raw_frames.append(extract_angles(result.pose_world_landmarks[0], result.pose_landmarks[0]))
            else:
                raw_frames.append(None)
    cap.release()

    valid = [f for f in raw_frames if f is not None]
    if len(valid) < MIN_VALID_FRAMES:
        return None

    first = next(i for i, f in enumerate(raw_frames) if f is not None)
    last  = len(raw_frames) - 1 - next(i for i, f in enumerate(reversed(raw_frames)) if f is not None)
    nan_frame = {feat: float("nan") for feat in FEATURES_BASE}
    cleaned = [f if f is not None else nan_frame for f in raw_frames[first:last + 1]]

    impact_idx = detect_impact_frame(cleaned)

    backswing_frames = max(1, impact_idx)
    downswing_frames = max(1, len(cleaned) - impact_idx)
    tempo_ratio = round(backswing_frames / downswing_frames, 3)

    return normalize_swing(cleaned, impact_idx, tempo_ratio), tempo_ratio


def compute_weights_for_club(swings: np.ndarray, club: str) -> dict:
    stds = np.nanstd(swings.reshape(-1, len(FEATURES)), axis=0)
    consistency = 1.0 / (stds + 1.0)
    consistency = consistency / consistency.mean()
    club_w = get_club_weights(club)
    return {feat: round(float(consistency[i] * club_w.get(feat, 1.0)), 4) for i, feat in enumerate(FEATURES)}


def build_model(swings: list[np.ndarray], tempo_ratios: list[float],
                camera_angles: set, n_players: int, club: str,
                video_sources: list[str]) -> dict:
    arr = np.stack(swings, axis=0)
    envelope = {}
    for fi, feat in enumerate(FEATURES):
        data = arr[:, :, fi]
        envelope[feat] = {
            "p25": [round(v, 4) for v in np.nanpercentile(data, 25, axis=0).tolist()],
            "p50": [round(v, 4) for v in np.nanpercentile(data, 50, axis=0).tolist()],
            "p75": [round(v, 4) for v in np.nanpercentile(data, 75, axis=0).tolist()],
        }

    tempo_arr = np.array(tempo_ratios)
    tempo = {
        "ratio_p25": round(float(np.percentile(tempo_arr, 25)), 3),
        "ratio_p50": round(float(np.percentile(tempo_arr, 50)), 3),
        "ratio_p75": round(float(np.percentile(tempo_arr, 75)), 3),
    }

    return {
        "version": "2.0",
        "club_category": club,
        "n_swings": len(swings),
        "n_frames": N_FRAMES,
        "impact_frame": IMPACT_FRAME,
        "features": FEATURES,
        "envelope": envelope,
        "weights": compute_weights_for_club(arr, club),
        "tempo": tempo,
        "metadata": {
            "built_at": str(date.today()),
            "camera_angles": list(camera_angles),
            "n_players": n_players,
            "video_sources": video_sources[:10],
        },
    }


def save_and_report(model: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model, indent=2))
    print(f"\n  Saved → {path}")
    print(f"  Club: {model['club_category']}  |  Swings: {model['n_swings']}  |  Players: {model['metadata']['n_players']}")
    print(f"  Tempo p50: {model['tempo']['ratio_p50']:.2f}  (p25: {model['tempo']['ratio_p25']:.2f}, p75: {model['tempo']['ratio_p75']:.2f})")
    top5 = sorted(model["weights"].items(), key=lambda x: -x[1])[:5]
    print(f"  Top weights: {', '.join(f'{f}={w:.2f}' for f, w in top5)}")


def run_club(club: str) -> bool:
    manifest_file = Path("pro_swings") / club / "manifest.json"
    if not manifest_file.exists():
        print(f"\n[skip] No manifest at {manifest_file} — run: python fetch_pro_swings.py --club {club}")
        return False

    manifest = json.loads(manifest_file.read_text())
    print(f"\n=== Building {club.upper().replace('_', ' ')} model ({len(manifest)} videos) ===")

    swings, tempo_ratios, players, angles, sources = [], [], set(), set(), []
    skipped = 0

    for entry in tqdm(manifest, desc=f"  {club}"):
        video_path = entry.get("file", "")
        if not Path(video_path).exists():
            tqdm.write(f"  [skip] not found: {video_path}")
            skipped += 1
            continue

        result = process_video(video_path)
        if result is None:
            tqdm.write(f"  [skip] insufficient pose data: {Path(video_path).name}")
            skipped += 1
            continue

        swing, tempo_ratio = result
        swings.append(swing)
        tempo_ratios.append(tempo_ratio)

        player = entry.get("player", "Unknown")
        if player != "Unknown":
            players.add(player)
        angles.add(entry.get("camera_angle", "unknown"))
        sources.append(entry.get("source_url", ""))
        tqdm.write(f"  [ok] {Path(video_path).name}  tempo={tempo_ratio:.2f}")

    print(f"\n  Processed: {len(swings)}  |  Skipped: {skipped}")

    if len(swings) < 2:
        print(f"  [warn] Only {len(swings)} valid swings — skipping (need ≥2)")
        return False

    model = build_model(swings, tempo_ratios, angles, len(players), club, sources)
    save_and_report(model, MODELS_DIR / f"{club}.json")
    return True


def main():
    parser = argparse.ArgumentParser(description="Build per-club reference models from downloaded videos")
    parser.add_argument(
        "--club", required=True,
        choices=CLUB_CATEGORIES + ["all"],
        help="Club category to build, or 'all'",
    )
    args = parser.parse_args()

    ensure_model()
    MODELS_DIR.mkdir(exist_ok=True)

    clubs = CLUB_CATEGORIES if args.club == "all" else [args.club]
    results = {}
    for club in clubs:
        results[club] = run_club(club)

    if args.club == "all":
        print(f"\n{'=' * 50}")
        print("  Model Build Summary")
        print(f"{'=' * 50}")
        for club, success in results.items():
            status = "✓" if success else "✗ (insufficient data)"
            print(f"  {club:<15} {status}")

    print(f"\nNext: python api_server.py\n")


if __name__ == "__main__":
    main()
