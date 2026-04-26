"""
Processes downloaded pro swing videos through MediaPipe Pose and builds the reference envelope.
Run after fetch_pro_swings.py: python build_reference_model.py

Uses the MediaPipe Tasks API (mediapipe >= 0.10) with the heavy pose landmarker model.
Output: reference_model.json — loaded by the browser app and api_server.py
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
from mediapipe.tasks.python.components.containers import landmark as mp_landmark

MANIFEST_FILE = Path("pro_swings/manifest.json")
OUTPUT_FILE = Path("reference_model.json")
MODEL_PATH = Path("pose_landmarker_heavy.task")
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"

N_FRAMES = 100
IMPACT_FRAME = 75
MIN_VALID_FRAMES = 20

FEATURES = [
    "lead_elbow",
    "trail_elbow",
    "lead_wrist_angle",
    "trail_wrist_angle",
    "lead_knee_flex",
    "trail_knee_flex",
    "spine_tilt",
    "hip_rotation",
    "shoulder_rotation",
    "x_factor",
    "hand_height",
]

# Landmark indices — identical to browser JS and MediaPipe Tasks spec
LM = {
    "LEFT_SHOULDER": 11, "RIGHT_SHOULDER": 12,
    "LEFT_ELBOW": 13,    "RIGHT_ELBOW": 14,
    "LEFT_WRIST": 15,    "RIGHT_WRIST": 16,
    "LEFT_HIP": 23,      "RIGHT_HIP": 24,
    "LEFT_KNEE": 25,     "RIGHT_KNEE": 26,
    "LEFT_ANKLE": 27,    "RIGHT_ANKLE": 28,
}


def ensure_model():
    if not MODEL_PATH.exists():
        print(f"Downloading pose landmarker model (~30MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print(f"  Saved to {MODEL_PATH}")


def angle_3pts(a, b, c) -> float:
    """Angle at vertex B formed by rays B→A and B→C, in degrees."""
    ba = np.array([a.x - b.x, a.y - b.y, a.z - b.z])
    bc = np.array([c.x - b.x, c.y - b.y, c.z - b.z])
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba < 1e-6 or norm_bc < 1e-6:
        return float("nan")
    cos_angle = np.clip(np.dot(ba, bc) / (norm_ba * norm_bc), -1.0, 1.0)
    return math.degrees(math.acos(cos_angle))


def is_left_handed(norm_lms) -> bool:
    """Left shoulder x < right shoulder x → golfer faces right → left-handed setup."""
    return norm_lms[LM["LEFT_SHOULDER"]].x < norm_lms[LM["RIGHT_SHOULDER"]].x


def extract_angles(world_lms, norm_lms) -> dict:
    """
    Extract 11 joint angle features from a single frame.
    Uses world landmarks (z is meaningful depth).
    Lead/trail terminology is handedness-agnostic.
    """
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

    lead_elbow       = angle_3pts(w[ls], w[le], w[lw])
    trail_elbow      = angle_3pts(w[rs], w[re], w[rw])
    lead_wrist_angle = angle_3pts(w[le], w[lw], w[ls])
    trail_wrist_angle= angle_3pts(w[re], w[rw], w[rs])
    lead_knee_flex   = angle_3pts(w[lh], w[lk], w[la])
    trail_knee_flex  = angle_3pts(w[rh], w[rk], w[ra])

    # Spine tilt: angle of mid-shoulder→mid-hip vector from vertical
    mid_sx = (w[ls].x + w[rs].x) / 2; mid_sy = (w[ls].y + w[rs].y) / 2; mid_sz = (w[ls].z + w[rs].z) / 2
    mid_hx = (w[lh].x + w[rh].x) / 2; mid_hy = (w[lh].y + w[rh].y) / 2; mid_hz = (w[lh].z + w[rh].z) / 2
    sv = np.array([mid_hx - mid_sx, mid_hy - mid_sy, mid_hz - mid_sz])
    sv_len = np.linalg.norm(sv)
    spine_tilt = float("nan") if sv_len < 1e-6 else math.degrees(math.acos(np.clip(sv[1] / sv_len, -1, 1)))

    # Rotation: arctan of z-depth delta between paired landmarks
    hip_rotation      = math.degrees(math.atan2(w[lh].z - w[rh].z, abs(w[lh].x - w[rh].x) + 1e-6))
    shoulder_rotation = math.degrees(math.atan2(w[ls].z - w[rs].z, abs(w[ls].x - w[rs].x) + 1e-6))

    # X-factor: shoulder separation from hips — key power metric
    x_factor = shoulder_rotation - hip_rotation

    # Hand height: avg wrist Y normalised within shoulder–hip Y range
    avg_wy   = (w[lw].y + w[rw].y) / 2
    shoulder_y = (w[ls].y + w[rs].y) / 2
    hip_y      = (w[lh].y + w[rh].y) / 2
    y_range    = abs(hip_y - shoulder_y)
    hand_height = float("nan") if y_range < 1e-6 else (shoulder_y - avg_wy) / y_range

    return {
        "lead_elbow": lead_elbow, "trail_elbow": trail_elbow,
        "lead_wrist_angle": lead_wrist_angle, "trail_wrist_angle": trail_wrist_angle,
        "lead_knee_flex": lead_knee_flex, "trail_knee_flex": trail_knee_flex,
        "spine_tilt": spine_tilt,
        "hip_rotation": hip_rotation, "shoulder_rotation": shoulder_rotation,
        "x_factor": x_factor, "hand_height": hand_height,
    }


def detect_impact_frame(frames: list[dict]) -> int:
    """
    Peak |Δlead_elbow| in the downswing (latter 60% of swing) = impact.
    """
    n = len(frames)
    if n < 4:
        return n // 2
    vals = [f.get("lead_elbow", float("nan")) for f in frames]
    velocities = []
    for i in range(1, n - 1):
        if not (math.isnan(vals[i-1]) or math.isnan(vals[i+1])):
            velocities.append((i, abs(vals[i+1] - vals[i-1])))
    downswing = [(i, v) for i, v in velocities if i >= int(n * 0.4)]
    if not downswing:
        return int(n * 0.75)
    return max(downswing, key=lambda x: x[1])[0]


def normalize_swing(frames: list[dict], impact_idx: int) -> np.ndarray:
    """
    Independently interpolate pre- and post-impact segments to N_FRAMES with
    impact always landing at IMPACT_FRAME. Returns (N_FRAMES, len(FEATURES)).
    """
    n = len(frames)
    normalized = np.full((N_FRAMES, len(FEATURES)), float("nan"))

    for fi, feat in enumerate(FEATURES):
        arr = np.array([f.get(feat, float("nan")) for f in frames])

        # Pre-impact: frames 0..impact_idx → output frames 0..IMPACT_FRAME
        pre = arr[:impact_idx + 1]
        valid_x = np.where(~np.isnan(pre))[0]
        if len(valid_x) >= 2:
            tgt = np.linspace(0, impact_idx, IMPACT_FRAME + 1)
            normalized[:IMPACT_FRAME + 1, fi] = np.interp(tgt, valid_x, pre[valid_x])

        # Post-impact: frames impact_idx..n-1 → output frames IMPACT_FRAME..N_FRAMES-1
        post = arr[impact_idx:]
        valid_x = np.where(~np.isnan(post))[0]
        if len(valid_x) >= 2:
            tgt = np.linspace(0, len(post) - 1, N_FRAMES - IMPACT_FRAME)
            normalized[IMPACT_FRAME:, fi] = np.interp(tgt, valid_x, post[valid_x])

    return normalized


def process_video(video_path: str, model_path: str) -> np.ndarray | None:
    """
    Run PoseLandmarker on every frame. Creates its own landmarker instance so
    timestamps always start from 0, satisfying VIDEO mode's monotonicity requirement.
    Returns normalised (N_FRAMES, n_features) or None.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    options = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    raw_frames = []
    frame_idx = 0

    with mp_vision.PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts_ms = int(frame_idx * (1000 / fps))
            result = landmarker.detect_for_video(mp_image, ts_ms)
            frame_idx += 1

            if not result.pose_world_landmarks or not result.pose_landmarks:
                raw_frames.append(None)
                continue

            angles = extract_angles(result.pose_world_landmarks[0], result.pose_landmarks[0])
            raw_frames.append(angles)

    cap.release()

    valid = [f for f in raw_frames if f is not None]
    if len(valid) < MIN_VALID_FRAMES:
        return None

    # Trim leading/trailing None frames
    first = next(i for i, f in enumerate(raw_frames) if f is not None)
    last  = len(raw_frames) - 1 - next(i for i, f in enumerate(reversed(raw_frames)) if f is not None)
    nan_frame = {feat: float("nan") for feat in FEATURES}
    cleaned = [f if f is not None else nan_frame for f in raw_frames[first:last + 1]]

    impact_idx = detect_impact_frame(cleaned)
    return normalize_swing(cleaned, impact_idx)


def compute_weights(swings: np.ndarray) -> dict:
    """
    Features that are MORE consistent across pros get HIGHER weights.
    Weight = 1 / (std + 1), normalised to mean = 1.
    """
    stds = np.nanstd(swings.reshape(-1, len(FEATURES)), axis=0)
    raw = 1.0 / (stds + 1.0)
    raw = raw / raw.mean()
    return {feat: round(float(raw[i]), 4) for i, feat in enumerate(FEATURES)}


def main():
    ensure_model()

    if not MANIFEST_FILE.exists():
        print(f"No manifest at {MANIFEST_FILE}. Run fetch_pro_swings.py first.")
        return

    manifest = json.loads(MANIFEST_FILE.read_text())
    print(f"\nProcessing {len(manifest)} videos with MediaPipe Pose (heavy model)...\n")

    all_swings = []
    n_players = set()
    camera_angles = set()
    skipped = 0

    for entry in tqdm(manifest, desc="Videos"):
        video_path = entry.get("file", "")
        if not Path(video_path).exists():
            tqdm.write(f"  [skip] not found: {video_path}")
            skipped += 1
            continue

        swing = process_video(video_path, str(MODEL_PATH))
        if swing is None:
            tqdm.write(f"  [skip] insufficient pose data: {Path(video_path).name}")
            skipped += 1
            continue

        all_swings.append(swing)
        if entry.get("player") and entry["player"] != "Unknown":
            n_players.add(entry["player"])
        if entry.get("camera_angle"):
            camera_angles.add(entry["camera_angle"])
        tqdm.write(f"  [ok] {Path(video_path).name}")

    if len(all_swings) < 2:
        print(f"\nNeed at least 2 valid swings. Got {len(all_swings)}.")
        return

    swings_array = np.stack(all_swings, axis=0)  # (n_swings, N_FRAMES, n_features)
    print(f"\nBuilding envelope from {len(all_swings)} swings ({skipped} skipped)...")

    envelope = {}
    for fi, feat in enumerate(FEATURES):
        data = swings_array[:, :, fi]
        envelope[feat] = {
            "p25": [round(v, 4) for v in np.nanpercentile(data, 25, axis=0).tolist()],
            "p50": [round(v, 4) for v in np.nanpercentile(data, 50, axis=0).tolist()],
            "p75": [round(v, 4) for v in np.nanpercentile(data, 75, axis=0).tolist()],
        }

    weights = compute_weights(swings_array)

    model = {
        "version": "1.0",
        "n_swings": len(all_swings),
        "n_frames": N_FRAMES,
        "impact_frame": IMPACT_FRAME,
        "features": FEATURES,
        "envelope": envelope,
        "weights": weights,
        "metadata": {
            "camera_angles": list(camera_angles),
            "n_players": len(n_players),
        },
    }

    OUTPUT_FILE.write_text(json.dumps(model, indent=2))
    print(f"\n=== Reference model saved to {OUTPUT_FILE} ===")
    print(f"  Swings: {len(all_swings)}")
    print(f"  Identified players: {len(n_players)}")
    print(f"  Camera angles: {camera_angles}")
    print(f"\nFeature weights:")
    for feat, w in sorted(weights.items(), key=lambda x: -x[1]):
        print(f"  {feat:<22} {w:.3f}")
    print("\nNext: python api_server.py\n")


if __name__ == "__main__":
    main()
