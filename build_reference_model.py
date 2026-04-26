"""
Processes downloaded pro swing videos through MediaPipe Pose and builds the reference envelopes.
Run after fetch_pro_swings.py: python build_reference_model.py

Outputs three files:
  reference_model.json         — combined (all clubs, used as fallback)
  reference_model_driver.json  — driver-specific envelope
  reference_model_iron.json    — iron-specific envelope
"""
import json
import math
import urllib.request
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

MANIFEST_FILE = Path("pro_swings/manifest.json")
MODEL_PATH    = Path("pose_landmarker_heavy.task")
MODEL_URL     = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"

N_FRAMES      = 100
IMPACT_FRAME  = 75
MIN_VALID_FRAMES = 20

FEATURES = [
    "lead_elbow", "trail_elbow",
    "lead_wrist_angle", "trail_wrist_angle",
    "lead_knee_flex", "trail_knee_flex",
    "spine_tilt", "hip_rotation", "shoulder_rotation",
    "x_factor", "hand_height",
]

LM = {
    "LEFT_SHOULDER": 11,  "RIGHT_SHOULDER": 12,
    "LEFT_ELBOW":    13,  "RIGHT_ELBOW":    14,
    "LEFT_WRIST":    15,  "RIGHT_WRIST":    16,
    "LEFT_HIP":      23,  "RIGHT_HIP":      24,
    "LEFT_KNEE":     25,  "RIGHT_KNEE":     26,
    "LEFT_ANKLE":    27,  "RIGHT_ANKLE":    28,
}


def detect_club_type(title: str, tagged: str | None) -> str:
    """Determine club type from manifest tag, falling back to title keyword search."""
    if tagged in ("driver", "iron"):
        return tagged
    t = title.lower()
    if any(k in t for k in ["iron", " 7i", " 6i", " 5i", " 4i", " 3i", "wedge", "approach"]):
        return "iron"
    # Default to driver for ambiguous titles
    return "driver"


def ensure_model():
    if not MODEL_PATH.exists():
        print("Downloading pose landmarker model (~30MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print(f"  Saved to {MODEL_PATH}")


def angle_3pts(a, b, c) -> float:
    """Angle at vertex B formed by rays B→A and B→C, in degrees."""
    ba = np.array([a.x - b.x, a.y - b.y, a.z - b.z])
    bc = np.array([c.x - b.x, c.y - b.y, c.z - b.z])
    nb, nc = np.linalg.norm(ba), np.linalg.norm(bc)
    if nb < 1e-6 or nc < 1e-6:
        return float("nan")
    return math.degrees(math.acos(np.clip(np.dot(ba, bc) / (nb * nc), -1.0, 1.0)))


def is_left_handed(norm_lms) -> bool:
    return norm_lms[LM["LEFT_SHOULDER"]].x < norm_lms[LM["RIGHT_SHOULDER"]].x


def extract_angles(world_lms, norm_lms) -> dict:
    """Extract all 11 features from one frame. Lead/trail is handedness-agnostic."""
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

    mid_sx = (w[ls].x + w[rs].x) / 2;  mid_sy = (w[ls].y + w[rs].y) / 2;  mid_sz = (w[ls].z + w[rs].z) / 2
    mid_hx = (w[lh].x + w[rh].x) / 2;  mid_hy = (w[lh].y + w[rh].y) / 2;  mid_hz = (w[lh].z + w[rh].z) / 2
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
    velocities = [(i, abs(vals[i+1] - vals[i-1]))
                  for i in range(1, n-1)
                  if not (math.isnan(vals[i-1]) or math.isnan(vals[i+1]))]
    downswing = [(i, v) for i, v in velocities if i >= int(n * 0.4)]
    return max(downswing, key=lambda x: x[1])[0] if downswing else int(n * 0.75)


def normalize_swing(frames: list[dict], impact_idx: int) -> np.ndarray:
    """Interpolate pre/post-impact independently to N_FRAMES, impact at IMPACT_FRAME."""
    normalized = np.full((N_FRAMES, len(FEATURES)), float("nan"))
    for fi, feat in enumerate(FEATURES):
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

    return normalized


def process_video(video_path: str) -> np.ndarray | None:
    """Run a fresh PoseLandmarker per video (timestamps reset each time)."""
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
                int(frame_idx * 1000 / fps)
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
    nan_frame = {feat: float("nan") for feat in FEATURES}
    cleaned = [f if f is not None else nan_frame for f in raw_frames[first:last + 1]]
    return normalize_swing(cleaned, detect_impact_frame(cleaned))


def compute_weights(swings: np.ndarray) -> dict:
    """Features more consistent across pros get higher weights."""
    stds = np.nanstd(swings.reshape(-1, len(FEATURES)), axis=0)
    raw  = 1.0 / (stds + 1.0)
    raw  = raw / raw.mean()
    return {feat: round(float(raw[i]), 4) for i, feat in enumerate(FEATURES)}


def build_envelope(swings: list[np.ndarray], camera_angles: set, n_players: int, club_type: str) -> dict:
    arr = np.stack(swings, axis=0)
    envelope = {}
    for fi, feat in enumerate(FEATURES):
        data = arr[:, :, fi]
        envelope[feat] = {
            "p25": [round(v, 4) for v in np.nanpercentile(data, 25, axis=0).tolist()],
            "p50": [round(v, 4) for v in np.nanpercentile(data, 50, axis=0).tolist()],
            "p75": [round(v, 4) for v in np.nanpercentile(data, 75, axis=0).tolist()],
        }
    return {
        "version": "2.0",
        "club_type": club_type,
        "n_swings": len(swings),
        "n_frames": N_FRAMES,
        "impact_frame": IMPACT_FRAME,
        "features": FEATURES,
        "envelope": envelope,
        "weights": compute_weights(arr),
        "metadata": {"camera_angles": list(camera_angles), "n_players": n_players},
    }


def save_and_report(model: dict, path: Path):
    path.write_text(json.dumps(model, indent=2))
    print(f"\n  Saved → {path}")
    print(f"  Swings: {model['n_swings']}  |  Players: {model['metadata']['n_players']}")
    print(f"  Weights (top 5):")
    top5 = sorted(model["weights"].items(), key=lambda x: -x[1])[:5]
    for feat, w in top5:
        print(f"    {feat:<22} {w:.3f}")


def main():
    ensure_model()

    if not MANIFEST_FILE.exists():
        print(f"No manifest at {MANIFEST_FILE}. Run fetch_pro_swings.py first.")
        return

    manifest = json.loads(MANIFEST_FILE.read_text())
    print(f"\nProcessing {len(manifest)} videos...\n")

    # Bucket results by club type
    swings_by_club: dict[str, list] = {"driver": [], "iron": []}
    players_by_club: dict[str, set] = {"driver": set(), "iron": set()}
    angles_by_club:  dict[str, set] = {"driver": set(), "iron": set()}
    all_swings, all_players, all_angles = [], set(), set()
    skipped = 0

    for entry in tqdm(manifest, desc="Videos"):
        video_path = entry.get("file", "")
        if not Path(video_path).exists():
            tqdm.write(f"  [skip] not found: {video_path}")
            skipped += 1
            continue

        club = detect_club_type(entry.get("title", ""), entry.get("club_type"))
        swing = process_video(video_path)
        if swing is None:
            tqdm.write(f"  [skip] insufficient pose data: {Path(video_path).name}")
            skipped += 1
            continue

        tqdm.write(f"  [ok] {Path(video_path).name}  [{club}]")
        swings_by_club[club].append(swing)
        all_swings.append(swing)

        player = entry.get("player", "Unknown")
        angle  = entry.get("camera_angle", "unknown")
        if player != "Unknown":
            players_by_club[club].add(player)
            all_players.add(player)
        angles_by_club[club].add(angle)
        all_angles.add(angle)

    print(f"\n{'='*50}")
    print(f"  Processed: {len(all_swings)}  |  Skipped: {skipped}")
    print(f"  Driver swings: {len(swings_by_club['driver'])}")
    print(f"  Iron swings:   {len(swings_by_club['iron'])}")
    print(f"{'='*50}")

    # Combined model (fallback)
    if len(all_swings) >= 2:
        print("\nBuilding combined model...")
        combined = build_envelope(all_swings, all_angles, len(all_players), "combined")
        save_and_report(combined, Path("reference_model.json"))

    # Driver model
    if len(swings_by_club["driver"]) >= 2:
        print("\nBuilding driver model...")
        driver_model = build_envelope(
            swings_by_club["driver"], angles_by_club["driver"],
            len(players_by_club["driver"]), "driver"
        )
        save_and_report(driver_model, Path("reference_model_driver.json"))
    else:
        print(f"\n[warn] Only {len(swings_by_club['driver'])} driver swings — skipping driver model (need ≥2)")

    # Iron model
    if len(swings_by_club["iron"]) >= 2:
        print("\nBuilding iron model...")
        iron_model = build_envelope(
            swings_by_club["iron"], angles_by_club["iron"],
            len(players_by_club["iron"]), "iron"
        )
        save_and_report(iron_model, Path("reference_model_iron.json"))
    else:
        print(f"\n[warn] Only {len(swings_by_club['iron'])} iron swings — skipping iron model (need ≥2)")

    print("\nNext: python api_server.py\n")


if __name__ == "__main__":
    main()
