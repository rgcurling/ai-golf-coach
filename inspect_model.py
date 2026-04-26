"""
Inspect and visualize the reference model envelope.
Usage: python inspect_model.py
Requires matplotlib: pip install matplotlib
"""
import json
import sys
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import numpy as np
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

MODEL_FILE = Path("reference_model.json")

PHASE_LABELS = {
    0:  "Address",
    20: "Takeaway",
    40: "Top",
    55: "Transition",
    75: "Impact",
    85: "Follow-through",
    99: "Finish",
}


def print_summary(model):
    print("\n" + "="*60)
    print(f"  Reference Model v{model['version']}")
    print("="*60)
    print(f"  Swings:          {model['n_swings']}")
    print(f"  Frames:          {model['n_frames']} (impact @ frame {model['impact_frame']})")
    print(f"  Players:         {model['metadata']['n_players']}")
    print(f"  Camera angles:   {', '.join(model['metadata']['camera_angles']) or 'mixed'}")

    print(f"\n{'Feature':<24} {'Weight':>7}  {'IQR @ Impact':>14}  {'P50 @ Impact':>14}")
    print("-"*64)

    env = model["envelope"]
    weights = model["weights"]
    impact = model["impact_frame"]

    rows = []
    for feat in model["features"]:
        p25 = env[feat]["p25"][impact]
        p50 = env[feat]["p50"][impact]
        p75 = env[feat]["p75"][impact]
        iqr = p75 - p25
        rows.append((feat, weights[feat], iqr, p50))

    for feat, w, iqr, p50 in sorted(rows, key=lambda x: -x[1]):
        unit = "" if feat == "hand_height" else "°"
        print(f"  {feat:<22} {w:>7.3f}  {iqr:>12.1f}{unit}  {p50:>12.1f}{unit}")

    print("\n  Higher weight = pros are more consistent = stricter scoring")


def print_keyframes(model):
    env = model["envelope"]
    print("\n" + "="*60)
    print("  P50 values at key swing phases")
    print("="*60)

    key_frames = sorted(PHASE_LABELS.keys())
    header = f"  {'Feature':<22}"
    for f in key_frames:
        header += f"  {PHASE_LABELS[f]:>12}"
    print(header)
    print("-" * (24 + 14 * len(key_frames)))

    for feat in model["features"]:
        unit = "" if feat == "hand_height" else "°"
        row = f"  {feat:<22}"
        for f in key_frames:
            val = env[feat]["p50"][f]
            row += f"  {val:>10.1f}{unit:1}"
        print(row)


def plot_envelopes(model):
    if not HAS_MPL:
        print("\nInstall matplotlib to see plots: pip install matplotlib")
        return

    features = model["features"]
    env = model["envelope"]
    n = model["n_frames"]
    impact = model["impact_frame"]
    x = np.arange(n)

    cols = 3
    rows = (len(features) + cols - 1) // cols
    fig = plt.figure(figsize=(16, rows * 3.2))
    fig.patch.set_facecolor("#1a3a2a")
    gs = gridspec.GridSpec(rows, cols, figure=fig, hspace=0.55, wspace=0.35)

    for i, feat in enumerate(features):
        ax = fig.add_subplot(gs[i // cols, i % cols])
        ax.set_facecolor("#152e22")

        p25 = np.array(env[feat]["p25"])
        p50 = np.array(env[feat]["p50"])
        p75 = np.array(env[feat]["p75"])

        ax.fill_between(x, p25, p75, alpha=0.35, color="#c9a84c", label="P25–P75")
        ax.plot(x, p50, color="#c9a84c", linewidth=1.8, label="P50")

        ax.axvline(impact, color="white", linewidth=1, linestyle="--", alpha=0.6)
        ax.text(impact + 1, ax.get_ylim()[1] if ax.get_ylim()[1] != 0 else 1,
                "impact", color="white", fontsize=6, alpha=0.7, va="top")

        for f, label in PHASE_LABELS.items():
            if f != impact:
                ax.axvline(f, color="#3d7a52", linewidth=0.5, linestyle=":", alpha=0.5)

        ax.set_title(feat.replace("_", " "), color="#faf7f2", fontsize=9, pad=4)
        ax.tick_params(colors="#9bb5a5", labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#3d7a52")
        ax.set_xlabel("frame", color="#9bb5a5", fontsize=7)
        unit = "" if feat == "hand_height" else "°"
        ax.set_ylabel(unit, color="#9bb5a5", fontsize=7)

    # Hide unused subplots
    for j in range(len(features), rows * cols):
        fig.add_subplot(gs[j // cols, j % cols]).set_visible(False)

    fig.suptitle(
        f"Pro Swing Reference Envelope  —  {model['n_swings']} swings  |  impact @ frame {impact}",
        color="#c9a84c", fontsize=12, fontfamily="serif", y=1.01
    )

    plt.savefig("reference_model_plot.png", dpi=150, bbox_inches="tight",
                facecolor="#1a3a2a")
    print(f"\nPlot saved to reference_model_plot.png")
    plt.show()


def main():
    if not MODEL_FILE.exists():
        print(f"reference_model.json not found. Run build_reference_model.py first.")
        sys.exit(1)

    model = json.loads(MODEL_FILE.read_text())

    print_summary(model)
    print_keyframes(model)
    plot_envelopes(model)


if __name__ == "__main__":
    main()
