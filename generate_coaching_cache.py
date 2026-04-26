import json
import os
import time

import anthropic
from dotenv import load_dotenv

load_dotenv()

FEATURE_DESCRIPTIONS = {
    "lead_elbow": "lead arm elbow flexion",
    "trail_elbow": "trail arm elbow position",
    "spine_tilt": "spine tilt angle at address and through swing",
    "hip_rotation": "hip rotation through the downswing",
    "x_factor": "shoulder to hip separation (X-Factor)",
    "hand_height": "hand height at the top of the backswing",
    "lead_knee_flex": "lead knee flex through the swing",
    "trail_knee_flex": "trail knee stability on the backswing",
    "shoulder_rotation": "shoulder turn and rotation",
    "lead_wrist_angle": "lead wrist position at the top",
    "trail_wrist_angle": "trail wrist hinge and retention",
}

SYSTEM_PROMPT = (
    "You are a PGA-certified golf instructor with 20 years of experience. "
    "Give one specific actionable coaching cue in a single sentence. "
    "Be direct and practical. Address the golfer as you. "
    "Do not use filler words. Make it something they can apply "
    "on the range immediately."
)

SEVERITIES = ["mild", "moderate", "severe"]
DIRECTIONS = ["high", "low"]

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

cache = {}
combinations = [
    (feat, direction, severity)
    for feat in FEATURE_DESCRIPTIONS
    for direction in DIRECTIONS
    for severity in SEVERITIES
]
total = len(combinations)

for i, (feature, direction, severity) in enumerate(combinations, 1):
    key = f"{feature}_{direction}_{severity}"
    user_message = (
        f"A golfer's {FEATURE_DESCRIPTIONS[feature]} is {direction} "
        f"({severity} deviation from professional tour baseline). "
        f"What is the single most important fix?"
    )

    for attempt in range(2):
        try:
            response = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=80,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            )
            cue = response.content[0].text.strip()
            cache[key] = cue
            print(f'[{i}/{total}] {key}: "{cue[:60]}..."')
            break
        except Exception as e:
            if attempt == 0:
                print(f"  [retry] {key}: {e}")
                time.sleep(2)
            else:
                print(f"  [failed] {key}: {e}")
                cache[key] = "Focus on this area during your next practice session."

    time.sleep(0.1)

os.makedirs("app", exist_ok=True)
with open("app/coaching_cache.json", "w") as f:
    json.dump(cache, f, indent=2)

print(f"\nTotal cues generated: {len(cache)}")
print(f"Estimated cost: $0.03")
print(f"Saved to app/coaching_cache.json")
