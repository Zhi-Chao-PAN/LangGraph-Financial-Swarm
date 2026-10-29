"""Offline scoring regressions; no model, network, or third-party dependencies."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List

from src.experiments.evaluate_metrics import (
    calculate_metrics,
    check_exact_match,
    score_answer,
)


class ScoringTests(unittest.TestCase):
    def test_historical_false_positives(self) -> None:
        cases = [
            ("Revenue was not 100; it was 200.", "100"),
            ("Revenue was 100 million USD.", "100 billion USD"),
            ("The loss was 100.", "-100"),
            ("100 million USD", "100 billion USD"),
            ("100", "-100"),
            ("100 USD", "100 EUR"),
            ("not profitable", "profitable"),
            ("2100", "100"),
        ]
        for answer, truth in cases:
            with self.subTest(answer=answer, truth=truth):
                self.assertFalse(check_exact_match(answer, truth))

    def test_complete_short_answers(self) -> None:
        for answer, truth in [
            ("-100.0", "-100"),
            ("1,000 USD", "1000 usd"),
            (" Yes ", "yes"),
            ("1e3", "1000"),
        ]:
            self.assertTrue(check_exact_match(answer, truth))
        self.assertFalse(check_exact_match("1.001", "1"))
        self.assertFalse(check_exact_match("10%", "10"))

    def test_prose_and_missing_answers_need_review(self) -> None:
        for answer in ["Revenue: 100", "100 or 200", "", None, True, float("nan")]:
            self.assertEqual(score_answer(answer, "100"), "review_required")
        self.assertEqual(
            score_answer("100", "Revenue in 2024 was 100"), "invalid_reference"
        )

    def test_counts_keep_review_rows_in_denominator(self) -> None:
        rows: List[Dict[str, Any]] = [
            {"model_answer": "100", "ground_truth": "100", "latency_s": 2},
            {"model_answer": "100 or 200", "ground_truth": "100"},
            {"model_answer": "200", "ground_truth": "100"},
            {"model_answer": "100", "ground_truth": "N/A"},
        ]
        result = calculate_metrics(rows)
        self.assertEqual(result["accuracy"], 0.25)
        self.assertEqual(result["review_required"], 1)
        self.assertEqual(result["invalid_reference"], 1)
        self.assertEqual(result["avg_latency"], 0.5)
        self.assertEqual(rows[1]["score_status"], "review_required")

    def test_explicit_final_answer_does_not_modify_response(self) -> None:
        rows: List[Dict[str, Any]] = [
            {
                "model_answer": "Intermediate value 100, result 200",
                "final_answer": "200",
                "ground_truth": "200",
            }
        ]
        self.assertEqual(calculate_metrics(rows)["accuracy"], 1)
        self.assertIn("Intermediate", rows[0]["model_answer"])

    def test_invalid_input_is_rejected(self) -> None:
        for data in [{}, [None], [{"latency_s": -1}], [{"latency_s": float("inf")}]]:
            with self.assertRaises(ValueError):
                calculate_metrics(data)
        self.assertEqual(calculate_metrics([])["total_samples"], 0)

    def test_cli_failure_status_and_output(self) -> None:
        script = (
            Path(__file__).resolve().parents[1] / "src/experiments/evaluate_metrics.py"
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "answers.data"
            for data in ["broken", "[]", "{}"]:
                path.write_text(data)
                run = subprocess.run(
                    [sys.executable, str(script), str(path)], capture_output=True
                )
                self.assertEqual(run.returncode, 2)
            path.write_text(
                json.dumps([{"model_answer": "100", "ground_truth": "100"}])
            )
            run = subprocess.run(
                [sys.executable, str(script), str(path)], capture_output=True
            )
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertTrue((Path(tmp) / "answers_scored.json").exists())
            self.assertNotIn("score_status", path.read_text())


if __name__ == "__main__":
    unittest.main()
