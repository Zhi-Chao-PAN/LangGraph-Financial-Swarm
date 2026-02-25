"""
Generate latency breakdown charts for paper.

Usage:
    python eval/plot_latency.py
"""

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)

RESULTS_DIR = Path(__file__).parent / "results" / "ablation"
OUTPUT_DIR = RESULTS_DIR


def load_analysis():
    path = RESULTS_DIR / "latency_analysis.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def plot_latency_by_mode(data: dict):
    """Bar chart: mean latency per ablation mode."""
    modes = list(data.keys())
    labels = {
        "full": "Full",
        "no_rewrite": "w/o Rewrite",
        "no_structured": "w/o Struct.",
        "no_multiquery": "w/o Multi-Q",
    }
    names = [labels.get(m, m) for m in modes]
    means = [data[m]["mean"] for m in modes]
    medians = [data[m]["median"] for m in modes]

    x = np.arange(len(modes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width / 2, means, width, label="Mean", color="#4C72B0")
    bars2 = ax.bar(x + width / 2, medians, width, label="Median", color="#55A868")

    ax.set_ylabel("Latency (seconds)")
    ax.set_title("Query Latency by Configuration")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=9)

    out = OUTPUT_DIR / "latency_by_mode.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved: {out}")


def plot_latency_by_type(data: dict):
    """Grouped bar chart: latency by question type across modes."""
    qtypes = ["single_hop", "comparison", "calculation", "multi_hop"]
    qtype_labels = ["Single-hop", "Comparison", "Calculation", "Multi-hop"]
    modes = [m for m in data if "by_type" in data[m]]
    labels = {
        "full": "Full",
        "no_rewrite": "w/o Rewrite",
        "no_structured": "w/o Struct.",
        "no_multiquery": "w/o Multi-Q",
    }

    x = np.arange(len(qtypes))
    width = 0.2
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, mode in enumerate(modes):
        vals = []
        for qt in qtypes:
            if qt in data[mode]["by_type"]:
                vals.append(data[mode]["by_type"][qt]["mean"])
            else:
                vals.append(0)
        ax.bar(x + i * width, vals, width, label=labels.get(mode, mode), color=colors[i % len(colors)])

    ax.set_ylabel("Mean Latency (seconds)")
    ax.set_title("Latency by Question Type")
    ax.set_xticks(x + width * (len(modes) - 1) / 2)
    ax.set_xticklabels(qtype_labels)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    out = OUTPUT_DIR / "latency_by_type.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved: {out}")


def plot_latency_by_difficulty(data: dict):
    """Grouped bar chart: latency by difficulty across modes."""
    diffs = ["easy", "medium", "hard"]
    diff_labels = ["Easy", "Medium", "Hard"]
    modes = [m for m in data if "by_difficulty" in data[m]]
    labels = {
        "full": "Full",
        "no_rewrite": "w/o Rewrite",
        "no_structured": "w/o Struct.",
        "no_multiquery": "w/o Multi-Q",
    }

    x = np.arange(len(diffs))
    width = 0.2
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, mode in enumerate(modes):
        vals = []
        for d in diffs:
            if d in data[mode]["by_difficulty"]:
                vals.append(data[mode]["by_difficulty"][d]["mean"])
            else:
                vals.append(0)
        ax.bar(x + i * width, vals, width, label=labels.get(mode, mode), color=colors[i % len(colors)])

    ax.set_ylabel("Mean Latency (seconds)")
    ax.set_title("Latency by Difficulty Level")
    ax.set_xticks(x + width * (len(modes) - 1) / 2)
    ax.set_xticklabels(diff_labels)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    out = OUTPUT_DIR / "latency_by_difficulty.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved: {out}")


def main():
    data = load_analysis()
    plot_latency_by_mode(data)
    plot_latency_by_type(data)
    plot_latency_by_difficulty(data)
    print("All latency charts generated.")


if __name__ == "__main__":
    main()
