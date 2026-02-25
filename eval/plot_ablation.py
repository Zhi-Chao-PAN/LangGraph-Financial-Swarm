"""
Generate ablation study comparison charts.
Reads results from eval/results/ablation/ and produces publication-ready figures.

Usage:
    python eval/plot_ablation.py
"""

import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")
import numpy as np

# Academic style settings
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

RESULTS_DIR = Path(__file__).parent / "results" / "ablation"
OUTPUT_DIR = Path(__file__).parent / "results" / "ablation"

MODE_LABELS = {
    "full": "Full System",
    "no_rewrite": "w/o Query\nRewriting",
    "no_structured": "w/o Structured\nChunking",
    "no_multiquery": "w/o Multi-Query\nSynthesis",
}

COLORS = {
    "full": "#2196F3",
    "no_rewrite": "#FF9800",
    "no_structured": "#F44336",
    "no_multiquery": "#9C27B0",
}


def load_results():
    data = {}
    for mode in MODE_LABELS:
        path = RESULTS_DIR / f"{mode}.json"
        if path.exists():
            with open(path, encoding="utf-8") as f:
                data[mode] = json.load(f)
    return data


def compute_metrics(results):
    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    avg_ps = sum(r.get("partial_score", 0) for r in results) / max(total, 1)
    hall = sum(1 for r in results if r["hallucination"])
    avg_lat = sum(r["latency_s"] for r in results) / max(total, 1)

    by_diff = {}
    for diff in ["easy", "medium", "hard"]:
        subset = [r for r in results if r.get("difficulty") == diff]
        if subset:
            by_diff[diff] = {
                "accuracy": sum(1 for r in subset if r["correct"]) / len(subset) * 100,
                "partial": sum(r.get("partial_score", 0) for r in subset) / len(subset),
            }

    return {
        "accuracy": correct / total * 100,
        "partial_score": avg_ps,
        "hallucination": hall / total * 100,
        "latency": avg_lat,
        "by_difficulty": by_diff,
    }


def plot_accuracy_comparison(all_metrics):
    """Bar chart: accuracy by mode."""
    modes = list(all_metrics.keys())
    accs = [all_metrics[m]["accuracy"] for m in modes]
    labels = [MODE_LABELS.get(m, m) for m in modes]
    colors = [COLORS.get(m, "#999") for m in modes]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, accs, color=colors, edgecolor="white", linewidth=1.5)

    for bar, acc in zip(bars, accs, strict=False):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"{acc:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=12)

    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Ablation Study: Component Contribution to Accuracy", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 110)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "ablation_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'ablation_accuracy.png'}")


def plot_difficulty_breakdown(all_metrics):
    """Grouped bar chart: accuracy by difficulty level."""
    modes = list(all_metrics.keys())
    diffs = ["easy", "medium", "hard"]
    x = np.arange(len(diffs))
    width = 0.8 / len(modes)

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, mode in enumerate(modes):
        accs = []
        for d in diffs:
            if d in all_metrics[mode]["by_difficulty"]:
                accs.append(all_metrics[mode]["by_difficulty"][d]["accuracy"])
            else:
                accs.append(0)
        offset = (i - len(modes) / 2 + 0.5) * width
        ax.bar(x + offset, accs, width, label=MODE_LABELS.get(mode, mode), color=COLORS.get(mode, "#999"), edgecolor="white")

    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Ablation: Accuracy by Difficulty Level", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([d.capitalize() for d in diffs], fontsize=12)
    ax.set_ylim(0, 115)
    ax.legend(loc="upper right", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "ablation_by_difficulty.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'ablation_by_difficulty.png'}")


def plot_partial_score(all_metrics):
    """Bar chart: average partial score by mode."""
    modes = list(all_metrics.keys())
    scores = [all_metrics[m]["partial_score"] for m in modes]
    labels = [MODE_LABELS.get(m, m) for m in modes]
    colors = [COLORS.get(m, "#999") for m in modes]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, scores, color=colors, edgecolor="white", linewidth=1.5)

    for bar, s in zip(bars, scores, strict=False):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{s:.3f}", ha="center", va="bottom", fontweight="bold", fontsize=12)

    ax.set_ylabel("Avg Partial Score", fontsize=12)
    ax.set_title("Ablation Study: Partial Score Comparison", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1.1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "ablation_partial_score.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'ablation_partial_score.png'}")


def main():
    data = load_results()
    if not data:
        print("No ablation results found. Run run_ablation.py first.")
        sys.exit(1)

    print(f"Found results for: {list(data.keys())}")

    all_metrics = {mode: compute_metrics(results) for mode, results in data.items()}

    # Print text summary
    print("\n" + "=" * 60)
    print("ABLATION STUDY SUMMARY")
    print("=" * 60)
    for mode, m in all_metrics.items():
        print(f"\n{MODE_LABELS.get(mode, mode)}:")
        print(f"  Accuracy:      {m['accuracy']:.1f}%")
        print(f"  Partial Score: {m['partial_score']:.3f}")
        print(f"  Hallucination: {m['hallucination']:.1f}%")
        print(f"  Avg Latency:   {m['latency']:.1f}s")
        for d, dm in m["by_difficulty"].items():
            print(f"  {d.capitalize():8s}:     {dm['accuracy']:.1f}% (ps={dm['partial']:.3f})")

    # Generate plots
    plot_accuracy_comparison(all_metrics)
    plot_difficulty_breakdown(all_metrics)
    plot_partial_score(all_metrics)

    print("\nDone! Charts saved to", OUTPUT_DIR)


if __name__ == "__main__":
    main()
