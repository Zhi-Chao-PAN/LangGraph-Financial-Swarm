"""Conservative exact matching of short final answers, not semantic grading."""

import argparse
import json
import math
import re
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Units must agree literally after case/whitespace normalization. No implicit
# conversion between million/billion, percent/fraction, or currencies is made.
NUMBER = r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][+-]?\d+)?"
VALUE = re.compile(rf"({NUMBER})(?:\s*(%|[a-zA-Z]+(?:\s+[a-zA-Z]+)*))?")
SCORER_VERSION = "short-answer-exact-v2"


def _text(value: object) -> str:
    if isinstance(value, bool) or value is None:
        return ""
    if not isinstance(value, (str, int, float)):
        return ""
    return " ".join(str(value).strip().casefold().split())


def _quantity(text: str) -> Optional[Tuple[Decimal, str]]:
    match = VALUE.fullmatch(text)
    if match is None:
        return None
    return Decimal(match[1].replace(",", "")), match[2] or ""


def normalize_number(value: object) -> Optional[Decimal]:
    """Parse only a complete, unitless numeric answer; never extract from prose."""
    quantity = _quantity(_text(value))
    return quantity[0] if quantity and not quantity[1] else None


def score_answer(model_answer: object, ground_truth: object) -> str:
    """Return match/mismatch/review_required/invalid_reference.

    Callers must supply a short final answer, with units matching the reference.
    Numeric prose is deliberately left for human review, even when it contains
    the expected number. This avoids counting negations and intermediate values.
    """
    answer, truth = _text(model_answer), _text(ground_truth)
    missing = {"", "n/a", "nan", "inf", "infinity", "-inf", "+inf"}
    if truth in missing:
        return "invalid_reference"
    if answer in missing:
        return "review_required"
    aq, tq = _quantity(answer), _quantity(truth)
    if tq is not None:
        if aq is None:
            return "review_required"
        return "match" if aq == tq else "mismatch"
    if re.search(r"\d", truth):
        return "invalid_reference"
    return "match" if answer == truth else "mismatch"


def check_exact_match(model_answer: object, ground_truth: object) -> bool:
    """Compatibility wrapper; unscorable answers are never reported as correct."""
    return score_answer(model_answer, ground_truth) == "match"


def calculate_metrics(results: object) -> Dict[str, Any]:
    if not isinstance(results, list) or any(
        not isinstance(row, dict) for row in results
    ):
        raise ValueError("input must be a JSON array of result objects")
    counts = {
        key: 0 for key in ("match", "mismatch", "review_required", "invalid_reference")
    }
    total_latency = 0.0
    for row in results:
        latency = row.get("latency_s", 0)
        if (
            isinstance(latency, bool)
            or not isinstance(latency, (int, float))
            or not math.isfinite(latency)
            or latency < 0
        ):
            raise ValueError("latency_s must be a finite, non-negative number")
        # Preserve the original response; only an explicit final_answer overrides it.
        status = score_answer(
            row.get("final_answer", row.get("model_answer")), row.get("ground_truth")
        )
        row.update(
            is_correct=status == "match",
            score_status=status,
            scorer_version=SCORER_VERSION,
        )
        counts[status] += 1
        total_latency += latency
    total = len(results)
    return {
        "scorer_version": SCORER_VERSION,
        "accuracy": counts["match"] / total if total else 0.0,
        "avg_latency": total_latency / total if total else 0.0,
        "total_samples": total,
        **counts,
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_file", type=Path)
    args = parser.parse_args(argv)
    try:
        data = json.loads(args.input_file.read_text(encoding="utf-8"))
        metrics = calculate_metrics(data)
        if not data:
            raise ValueError("no evaluation samples")
        print(json.dumps(metrics, indent=2))
        output = args.input_file.with_name(args.input_file.stem + "_scored.json")
        output.write_text(
            json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(f"Scored results saved to: {output}")
        return 0
    except (OSError, ValueError, TypeError) as error:
        print(f"Error calculating metrics: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
