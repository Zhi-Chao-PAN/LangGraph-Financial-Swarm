"""
Latency breakdown analysis for evaluation results.

Usage:
    python eval/analyze_latency.py
"""

import json
import statistics
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results" / "ablation"


def analyze(mode: str, results: list[dict]) -> dict:
    """Compute latency statistics for a result set."""
    latencies = [r["latency_s"] for r in results]
    by_diff = {}
    for diff in ["easy", "medium", "hard"]:
        subset = [r["latency_s"] for r in results if r.get("difficulty") == diff]
        if subset:
            by_diff[diff] = {
                "mean": statistics.mean(subset),
                "median": statistics.median(subset),
                "p95": sorted(subset)[int(len(subset) * 0.95)] if len(subset) >= 2 else subset[0],
                "count": len(subset),
            }

    by_type = {}
    for qtype in ["single_hop", "comparison", "calculation", "multi_hop"]:
        subset = [r["latency_s"] for r in results if r.get("type") == qtype]
        if subset:
            by_type[qtype] = {
                "mean": statistics.mean(subset),
                "median": statistics.median(subset),
                "count": len(subset),
            }

    return {
        "mode": mode,
        "total": len(latencies),
        "mean": statistics.mean(latencies),
        "median": statistics.median(latencies),
        "stdev": statistics.stdev(latencies) if len(latencies) > 1 else 0,
        "min": min(latencies),
        "max": max(latencies),
        "p95": sorted(latencies)[int(len(latencies) * 0.95)] if len(latencies) >= 2 else latencies[0],
        "by_difficulty": by_diff,
        "by_type": by_type,
    }


def main():
    all_stats = {}
    for path in sorted(RESULTS_DIR.glob("*.json")):
        if path.suffix != ".json" or path.name.endswith(".tex"):
            continue
        mode = path.stem
        with open(path, encoding="utf-8") as f:
            results = json.load(f)
        stats = analyze(mode, results)
        all_stats[mode] = stats

    if not all_stats:
        print("No results found.")
        return

    # Print summary table
    print(f"{'Mode':<20} {'N':>4} {'Mean':>8} {'Median':>8} {'P95':>8} {'Min':>8} {'Max':>8}")
    print("-" * 72)
    for mode, s in all_stats.items():
        print(
            f"{mode:<20} {s['total']:>4} {s['mean']:>8.1f} {s['median']:>8.1f}"
            f" {s['p95']:>8.1f} {s['min']:>8.1f} {s['max']:>8.1f}"
        )

    # Breakdown by difficulty
    print("\nBy Difficulty:")
    print(f"{'Mode':<20} {'Easy':>10} {'Medium':>10} {'Hard':>10}")
    print("-" * 55)
    for mode, s in all_stats.items():
        cells = []
        for diff in ["easy", "medium", "hard"]:
            if diff in s["by_difficulty"]:
                cells.append(f"{s['by_difficulty'][diff]['mean']:.1f}s")
            else:
                cells.append("--")
        print(f"{mode:<20} {cells[0]:>10} {cells[1]:>10} {cells[2]:>10}")

    # Breakdown by type
    print("\nBy Question Type:")
    print(f"{'Mode':<20} {'single_hop':>12} {'comparison':>12} {'calculation':>12} {'multi_hop':>12}")
    print("-" * 75)
    for mode, s in all_stats.items():
        cells = []
        for qtype in ["single_hop", "comparison", "calculation", "multi_hop"]:
            if qtype in s["by_type"]:
                cells.append(f"{s['by_type'][qtype]['mean']:.1f}s")
            else:
                cells.append("--")
        print(f"{mode:<20} {cells[0]:>12} {cells[1]:>12} {cells[2]:>12} {cells[3]:>12}")

    # Save JSON
    out_path = RESULTS_DIR / "latency_analysis.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
