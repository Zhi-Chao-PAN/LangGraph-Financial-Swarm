"""
Generate LaTeX tables from ablation results for paper inclusion.

Usage:
    python eval/gen_latex.py
"""

import json
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results" / "ablation"

MODE_LABELS = {
    "full": "Full System",
    "no_rewrite": "w/o Query Rewriting",
    "no_structured": "w/o Structured Chunking",
    "no_multiquery": "w/o Multi-Query Synthesis",
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
                "correct": sum(1 for r in subset if r["correct"]),
                "total": len(subset),
                "accuracy": sum(1 for r in subset if r["correct"]) / len(subset) * 100,
            }

    by_type = {}
    for qtype in ["single_hop", "comparison", "calculation", "multi_hop"]:
        subset = [r for r in results if r.get("type") == qtype]
        if subset:
            by_type[qtype] = {
                "correct": sum(1 for r in subset if r["correct"]),
                "total": len(subset),
                "accuracy": sum(1 for r in subset if r["correct"]) / len(subset) * 100,
            }

    return {
        "accuracy": correct / total * 100,
        "correct": correct,
        "total": total,
        "partial_score": avg_ps,
        "hallucination": hall / total * 100,
        "latency": avg_lat,
        "by_difficulty": by_diff,
        "by_type": by_type,
    }


def gen_main_table(all_metrics):
    """Generate main ablation comparison table."""
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Ablation Study Results (40-question benchmark)}",
        r"\label{tab:ablation}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"\textbf{Configuration} & \textbf{Accuracy} & \textbf{Partial Score} & \textbf{Hallucination} & \textbf{Latency (s)} \\",
        r"\midrule",
    ]

    for mode, m in all_metrics.items():
        label = MODE_LABELS.get(mode, mode)
        acc = f"{m['accuracy']:.1f}\\% ({m['correct']}/{m['total']})"
        ps = f"{m['partial_score']:.3f}"
        hall = f"{m['hallucination']:.1f}\\%"
        lat = f"{m['latency']:.1f}"
        lines.append(f"{label} & {acc} & {ps} & {hall} & {lat} \\\\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def gen_difficulty_table(all_metrics):
    """Generate accuracy by difficulty table."""
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Accuracy by Difficulty Level}",
        r"\label{tab:difficulty}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"\textbf{Configuration} & \textbf{Easy} & \textbf{Medium} & \textbf{Hard} \\",
        r"\midrule",
    ]

    for mode, m in all_metrics.items():
        label = MODE_LABELS.get(mode, mode)
        cells = []
        for diff in ["easy", "medium", "hard"]:
            if diff in m["by_difficulty"]:
                d = m["by_difficulty"][diff]
                cells.append(f"{d['correct']}/{d['total']} ({d['accuracy']:.0f}\\%)")
            else:
                cells.append("--")
        lines.append(f"{label} & {' & '.join(cells)} \\\\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def gen_type_table(all_metrics):
    """Generate accuracy by question type table."""
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Accuracy by Question Type}",
        r"\label{tab:qtype}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"\textbf{Configuration} & \textbf{Single-hop} & \textbf{Comparison} & \textbf{Calculation} & \textbf{Multi-hop} \\",
        r"\midrule",
    ]

    for mode, m in all_metrics.items():
        label = MODE_LABELS.get(mode, mode)
        cells = []
        for qtype in ["single_hop", "comparison", "calculation", "multi_hop"]:
            if qtype in m["by_type"]:
                t = m["by_type"][qtype]
                cells.append(f"{t['correct']}/{t['total']}")
            else:
                cells.append("--")
        lines.append(f"{label} & {' & '.join(cells)} \\\\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def main():
    data = load_results()
    if not data:
        print("No results found.")
        return

    all_metrics = {mode: compute_metrics(results) for mode, results in data.items()}

    output_dir = RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    tables = {
        "ablation_main.tex": gen_main_table(all_metrics),
        "ablation_difficulty.tex": gen_difficulty_table(all_metrics),
        "ablation_qtype.tex": gen_type_table(all_metrics),
    }

    for filename, content in tables.items():
        path = output_dir / filename
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Saved: {path}")
        print(content)
        print()


if __name__ == "__main__":
    main()
