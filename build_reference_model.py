"""
Processes downloaded pro swing videos through MediaPipe Pose and builds the reference envelope.
Run after fetch_pro_swings.py: python build_reference_model.py

Output: reference_model.json — loaded by the browser app and api_server.py
"""
import json
import math
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

MANIFEST_FILE = Path("pro_swings/manifest.json")
OUTPUT_FILE = Path("reference_model.json")
N_FRAMES = 100       # every swing is normalized to this many frames
IMPACT_FRAME = 75    # impact is always placed here (0-indexed)
MIN_VALID_FRAMES = 20
MODEL_COMPLEXITY = 2

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

# MediaPipe landmark indices
LM = mp.solutions.pose.PoseLandmark


def angle_3pts(a, b, c) -> float:
    """
    Returns the angle at vertex B formed by rays B→A and B→C, in degrees.
    Works with (x, y, z) tuples or any 3-element sequence.
    """
    ba = np.array([a[0] - b[0], a[1] - b[1], a[2] - b[2]])
    bc = np.array([c[0] - b[0], c[1] - b[1], c[2] - b[2]])
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba < 1e-6 or norm_bc < 1e-6:
        return float("nan")
    cos_angle = np.clip(np.dot(ba, bc) / (norm_ba * norm_bc), -1.0, 1.0)
    return math.degrees(math.acos(cos_angle))


def is_left_handed(lms) -> bool:
    """
    Detect handedness from the golfer's setup pose.
    In a right-handed address, the right shoulder is deeper (larger z) than left
    when the golfer faces the camera (face-on view).
    For down-the-line, we fall back to assuming right-handed.
    """
    left_shoulder = lms[LM.LEFT_SHOULDER.value]
    right_shoulder = lms[LM.RIGHT_SHOULDER.value]
    # If left shoulder is to the right of right shoulder in x, golfer faces right→ face-on left-handed
    return left_shoulder.x < right_shoulder.x


def extract_angles(lms, left_handed: bool) -> dict:
    """
    Extract the 11 feature angles from a single frame's landmarks.
    All angles use WORLD coordinates (z depth is meaningful).

    Convention: for a right-handed golfer the lead arm is the LEFT arm.
    We flip indices for left-handed golfers so downstream code is always
    in "lead/trail" terms regardless of handedness.
    """
    def pt(idx):
        l = lms[idx]
        return (l.x, l.y, l.z)

    if left_handed:
        # Swap left/right so "lead" always means the target-side arm/leg
        ls, rs = LM.RIGHT_SHOULDER.value, LM.LEFT_SHOULDER.value
        le, re = LM.RIGHT_ELBOW.value, LM.LEFT_ELBOW.value
        lw, rw = LM.RIGHT_WRIST.value, LM.LEFT_WRIST.value
        lh, rh = LM.RIGHT_HIP.value, LM.LEFT_HIP.value
        lk, rk = LM.RIGHT_KNEE.value, LM.LEFT_KNEE.value
        la, ra = LM.RIGHT_ANKLE.value, LM.LEFT_ANKLE.value
    else:
        ls, rs = LM.LEFT_SHOULDER.value, LM.RIGHT_SHOULDER.value
        le, re = LM.LEFT_ELBOW.value, LM.RIGHT_ELBOW.value
        lw, rw = LM.LEFT_WRIST.value, LM.RIGHT_WRIST.value
        lh, rh = LM.LEFT_HIP.value, LM.RIGHT_HIP.value
        lk, rk = LM.LEFT_KNEE.value, LM.RIGHT_KNEE.value
        la, ra = LM.LEFT_ANKLE.value, LM.RIGHT_ANKLE.value

    lead_elbow = angle_3pts(pt(ls), pt(le), pt(lw))
    trail_elbow = angle_3pts(pt(rs), pt(re), pt(rw))
    lead_wrist_angle = angle_3pts(pt(le), pt(lw), pt(ls))
    trail_wrist_angle = angle_3pts(pt(re), pt(rw), pt(rs))
    lead_knee_flex = angle_3pts(pt(lh), pt(lk), pt(la))
    trail_knee_flex = angle_3pts(pt(rh), pt(rk), pt(ra))

    # Spine tilt: angle of mid-shoulder to mid-hip vector from vertical (0° = upright)
    mid_shoulder = np.array([(pt(ls)[i] + pt(rs)[i]) / 2 for i in range(3)])
    mid_hip = np.array([(pt(lh)[i] + pt(rh)[i]) / 2 for i in range(3)])
    spine_vec = mid_hip - mid_shoulder
    vertical = np.array([0, 1, 0])
    norm_sv = np.linalg.norm(spine_vec)
    if norm_sv < 1e-6:
        spine_tilt = float("nan")
    else:
        spine_tilt = math.degrees(math.acos(np.clip(np.dot(spine_vec / norm_sv, vertical), -1, 1)))

    # Hip/shoulder rotation: arctan of z-depth delta between paired landmarks.
    # Positive = open (rotated toward target), negative = closed.
    hip_rotation = math.degrees(math.atan2(pt(lh)[2] - pt(rh)[2], abs(pt(lh)[0] - pt(rh)[0]) + 1e-6))
    shoulder_rotation = math.degrees(math.atan2(pt(ls)[2] - pt(rs)[2], abs(pt(ls)[0] - pt(rs)[0]) + 1e-6))

    # X-factor: shoulder rotation relative to hip rotation. Key power metric.
    x_factor = shoulder_rotation - hip_rotation

    # Hand height: avg wrist Y normalized to [0,1] within shoulder-hip Y range
    avg_wrist_y = (pt(lw)[1] + pt(rw)[1]) / 2
    shoulder_y = (pt(ls)[1] + pt(rs)[1]) / 2
    hip_y = (pt(lh)[1] + pt(rh)[1]) / 2
    y_range = abs(hip_y - shoulder_y)
    if y_range < 1e-6:
        hand_height = float("nan")
    else:
        hand_height = (shoulder_y - avg_wrist_y) / y_range  # positive = above shoulders

    return {
        "lead_elbow": lead_elbow,
        "trail_elbow": trail_elbow,
        "lead_wrist_angle": lead_wrist_angle,
        "trail_wrist_angle": trail_wrist_angle,
        "lead_knee_flex": lead_knee_flex,
        "trail_knee_flex": trail_knee_flex,
        "spine_tilt": spine_tilt,
        "hip_rotation": hip_rotation,
        "shoulder_rotation": shoulder_rotation,
        "x_factor": x_factor,
        "hand_height": hand_height,
    }


def detect_impact_frame(frames: list[dict]) -> int:
    """
    Finds the impact frame index using peak wrist velocity in the downswing.
    We look for the maximum speed of the average wrist position — this corresponds
    to the moment of maximum club head speed, which coincides with ball contact.
    We search only the second half of the swing (downswing phase).
    """
    n = len(frames)
    if n < 4:
        return n // 2

    # Build wrist velocity signal from lead wrist angle changes as a proxy
    # (actual x/y positions aren't directly available here, we use angle derivative)
    lead_elbows = [f.get("lead_elbow", float("nan")) for f in frames]
    valid = [(i, v) for i, v in enumerate(lead_elbows) if not math.isnan(v)]
    if len(valid) < 4:
        return n // 2

    # Compute finite-difference velocity on the lead_elbow angle.
    # At impact the arm is straightening fastest → peak positive derivative.
    velocities = []
    for i in range(1, len(valid) - 1):
        prev_val = valid[i - 1][1]
        next_val = valid[i + 1][1]
        velocities.append((valid[i][0], abs(next_val - prev_val)))

    # Only consider the latter 60% of the swing (downswing portion)
    start_search = int(n * 0.4)
    downswing_vels = [(idx, vel) for idx, vel in velocities if idx >= start_search]
    if not downswing_vels:
        return int(n * 0.75)

    impact_idx = max(downswing_vels, key=lambda x: x[1])[0]
    return impact_idx


def normalize_swing(frames: list[dict], impact_idx: int) -> np.ndarray:
    """
    Linearly interpolates the swing to exactly N_FRAMES frames with impact at IMPACT_FRAME.
    Pre-impact and post-impact segments are interpolated independently to preserve timing.

    Returns shape (N_FRAMES, len(FEATURES)) — NaN for frames with missing landmarks.
    """
    n = len(frames)
    feat_arrays = {f: np.array([frame.get(f, float("nan")) for frame in frames]) for f in FEATURES}

    pre_frames = impact_idx + 1       # frames 0..impact_idx inclusive
    post_frames = n - impact_idx      # frames impact_idx..n-1 inclusive

    norm_pre = IMPACT_FRAME + 1       # target pre-impact length (0..IMPACT_FRAME)
    norm_post = N_FRAMES - IMPACT_FRAME  # target post-impact length (IMPACT_FRAME..N_FRAMES-1)

    normalized = np.full((N_FRAMES, len(FEATURES)), float("nan"))

    for fi, feat in enumerate(FEATURES):
        arr = feat_arrays[feat]

        # Pre-impact interpolation
        pre_src = np.linspace(0, impact_idx, norm_pre)
        pre_dst = np.arange(norm_pre)
        # Only interpolate over non-NaN points
        valid_mask = ~np.isnan(arr[:impact_idx + 1])
        if valid_mask.sum() >= 2:
            valid_x = np.arange(impact_idx + 1)[valid_mask]
            valid_y = arr[:impact_idx + 1][valid_mask]
            normalized[:norm_pre, fi] = np.interp(pre_src, valid_x, valid_y)

        # Post-impact interpolation
        post_src = np.linspace(impact_idx, n - 1, norm_post)
        post_dst = np.arange(IMPACT_FRAME, N_FRAMES)
        valid_mask = ~np.isnan(arr[impact_idx:])
        if valid_mask.sum() >= 2:
            valid_x = np.arange(n - impact_idx)[valid_mask] + impact_idx
            valid_y = arr[impact_idx:][valid_mask]
            normalized[IMPACT_FRAME:, fi] = np.interp(post_src, valid_x, valid_y)

    return normalized


def process_video(video_path: str) -> np.ndarray | None:
    """
    Runs MediaPipe Pose on every frame of a video.
    Returns normalized swing array (N_FRAMES, len(FEATURES)) or None if insufficient data.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    mp_pose = mp.solutions.pose.Pose(
        model_complexity=MODEL_COMPLEXITY,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        static_image_mode=False,
    )

    raw_frames = []
    left_handed = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = mp_pose.process(rgb)

        if result.pose_world_landmarks is None:
            raw_frames.append(None)
            continue

        lms = result.pose_world_landmarks.landmark

        if left_handed is None:
            left_handed = is_left_handed(result.pose_landmarks.landmark)

        angles = extract_angles(lms, left_handed)
        raw_frames.append(angles)

    cap.release()
    mp_pose.close()

    # Filter out leading/trailing None frames; keep only inner None as gaps
    valid_frames = [f for f in raw_frames if f is not None]
    if len(valid_frames) < MIN_VALID_FRAMES:
        return None

    # Rebuild list preserving None gaps for timing, strip outer Nones
    first_valid = next(i for i, f in enumerate(raw_frames) if f is not None)
    last_valid = len(raw_frames) - 1 - next(i for i, f in enumerate(reversed(raw_frames)) if f is not None)
    trimmed = raw_frames[first_valid:last_valid + 1]

    # Fill None gaps with NaN dicts
    nan_frame = {f: float("nan") for f in FEATURES}
    cleaned = [f if f is not None else nan_frame for f in trimmed]

    impact_idx = detect_impact_frame(cleaned)
    return normalize_swing(cleaned, impact_idx)


def compute_weights(swings: np.ndarray) -> dict:
    """
    Features with LOW variance across pros are HIGHLY consistent → higher weight.
    Weight = 1 / (std + epsilon), normalized so weights sum to len(FEATURES).
    """
    # swings shape: (n_swings, N_FRAMES, n_features)
    # Collapse time axis: use mean std over all frames
    stds = np.nanstd(swings.reshape(-1, len(FEATURES)), axis=0)
    raw_weights = 1.0 / (stds + 1.0)  # +1 for stability, prevents div by near-zero
    # Normalize so mean weight == 1.0
    raw_weights = raw_weights / raw_weights.mean()
    return {feat: round(float(raw_weights[i]), 4) for i, feat in enumerate(FEATURES)}


def main():
    if not MANIFEST_FILE.exists():
        print(f"No manifest found at {MANIFEST_FILE}. Run fetch_pro_swings.py first.")
        return

    manifest = json.loads(MANIFEST_FILE.read_text())
    print(f"\nProcessing {len(manifest)} videos with MediaPipe Pose (complexity={MODEL_COMPLEXITY})...\n")

    all_swings = []
    n_players = set()
    camera_angles = set()
    skipped = 0

    for entry in tqdm(manifest, desc="Videos"):
        video_path = entry.get("file", "")
        if not Path(video_path).exists():
            tqdm.write(f"  [skip] File not found: {video_path}")
            skipped += 1
            continue

        swing = process_video(video_path)
        if swing is None:
            tqdm.write(f"  [skip] Insufficient pose data: {video_path}")
            skipped += 1
            continue

        all_swings.append(swing)
        if entry.get("player") and entry["player"] != "Unknown":
            n_players.add(entry["player"])
        if entry.get("camera_angle"):
            camera_angles.add(entry["camera_angle"])

    if len(all_swings) < 2:
        print(f"\nNeed at least 2 valid swings to build envelope. Got {len(all_swings)}.")
        print("Try running fetch_pro_swings.py again or check that yt-dlp downloaded videos.")
        return

    swings_array = np.stack(all_swings, axis=0)  # (n_swings, N_FRAMES, n_features)
    print(f"\nBuilding envelope from {len(all_swings)} swings ({skipped} skipped)...")

    # Per-feature per-frame percentiles
    envelope = {}
    for fi, feat in enumerate(FEATURES):
        feat_data = swings_array[:, :, fi]  # (n_swings, N_FRAMES)
        p25 = np.nanpercentile(feat_data, 25, axis=0).tolist()
        p50 = np.nanpercentile(feat_data, 50, axis=0).tolist()
        p75 = np.nanpercentile(feat_data, 75, axis=0).tolist()
        envelope[feat] = {
            "p25": [round(v, 4) for v in p25],
            "p50": [round(v, 4) for v in p50],
            "p75": [round(v, 4) for v in p75],
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
    print(f"\nFeature weights (higher = pros are more consistent here):")
    for feat, w in sorted(weights.items(), key=lambda x: -x[1]):
        print(f"  {feat:<22} {w:.3f}")
    print("\nNext: python api_server.py\n")


if __name__ == "__main__":
    main()
