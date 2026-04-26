"""
Generates per-club coaching cues for all feature/direction/severity combinations.
Produces 396 cues (6 clubs × 11 features × 2 directions × 3 severities).
Run once; resumes from checkpoint if interrupted.

Usage: python generate_coaching_cache.py
"""
import json
import os
import time
from pathlib import Path

import anthropic
from dotenv import load_dotenv

load_dotenv()

CLUB_CATEGORIES = ["driver", "fairway_wood", "long_iron", "mid_iron", "short_iron", "wedge"]

FEATURE_DESCRIPTIONS = {
    "lead_elbow": "lead arm elbow flexion",
    "trail_elbow": "trail arm elbow position",
    "spine_tilt": "spine tilt angle",
    "hip_rotation": "hip rotation through the swing",
    "x_factor": "shoulder to hip separation (X-Factor)",
    "hand_height": "hand height at the top of the backswing",
    "lead_knee_flex": "lead knee flex through the swing",
    "trail_knee_flex": "trail knee stability",
    "shoulder_rotation": "shoulder turn and rotation",
    "lead_wrist_angle": "lead wrist position at the top",
    "trail_wrist_angle": "trail wrist hinge and retention",
}

SEVERITIES  = ["mild", "moderate", "severe"]
DIRECTIONS  = ["high", "low"]
CACHE_PATH  = Path("app/coaching_cache.json")

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Load existing cache to support resuming
cache: dict = {}
if CACHE_PATH.exists():
    try:
        cache = json.loads(CACHE_PATH.read_text())
        print(f"Resuming — {len(cache)} cues already cached")
    except json.JSONDecodeError:
        pass

combinations = [
    (club, feat, direction, severity)
    for club       in CLUB_CATEGORIES
    for feat       in FEATURE_DESCRIPTIONS
    for direction  in DIRECTIONS
    for severity   in SEVERITIES
]
total = len(combinations)
print(f"Total combinations: {total}  ({len(CLUB_CATEGORIES)} clubs × {len(FEATURE_DESCRIPTIONS)} features × {len(DIRECTIONS)} directions × {len(SEVERITIES)} severities)")

generated = skipped = 0

for i, (club, feature, direction, severity) in enumerate(combinations, 1):
    key = f"{club}_{feature}_{direction}_{severity}"
    if key in cache:
        skipped += 1
        continue

    club_display = club.replace("_", " ")
    system_prompt = (
        f"You are a PGA-certified golf instructor specializing in {club_display} technique. "
        f"Give one specific actionable coaching cue in a single sentence for a {club_display} swing issue. "
        "Be direct and practical. Address the golfer as you. "
        "Do not use filler words. Make it something they can apply on the range immediately."
    )
    user_message = (
        f"A golfer's {FEATURE_DESCRIPTIONS[feature]} is {direction} "
        f"({severity} deviation from professional tour baseline) during their {club_display} swing. "
        f"What is the single most important fix for this specific club?"
    )

    for attempt in range(2):
        try:
            response = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=80,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
            )
            cue = response.content[0].text.strip()
            cache[key] = cue
            generated += 1
            print(f'[{i}/{total}] {key}: "{cue[:60]}..."')
            break
        except Exception as e:
            if attempt == 0:
                print(f"  [retry] {key}: {e}")
                time.sleep(2)
            else:
                print(f"  [failed] {key}: {e}")
                cache[key] = "Focus on this area during your next practice session."
                generated += 1

    time.sleep(0.1)

    # Checkpoint every 50 new cues
    if generated % 50 == 0 and generated > 0:
        CACHE_PATH.parent.mkdir(exist_ok=True)
        CACHE_PATH.write_text(json.dumps(cache, indent=2))
        print(f"  [checkpoint] {len(cache)} cues saved")

CACHE_PATH.parent.mkdir(exist_ok=True)
CACHE_PATH.write_text(json.dumps(cache, indent=2))

print(f"\nTotal cues: {len(cache)}  (generated: {generated}, skipped: {skipped})")
print(f"Estimated cost: ~${generated * 0.00003:.4f}")
print(f"Saved to {CACHE_PATH}")
print("\nFallback key format: {feature}_{direction}_{severity}")
print("Club-specific key format: {club}_{feature}_{direction}_{severity}")
