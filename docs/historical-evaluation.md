# Historical evaluation claims — not a validated current result

The following section is preserved from the previous README for traceability.
The previous scorer matched any number or substring in the response, including
negated values and mismatched units. These figures must not be cited as validated
performance. Re-run saved per-answer outputs under a documented rubric before
publishing replacement results. No new model evaluation was run for this fix.

## 📊 Evaluation (Preliminary)

We compared the Swarm architecture against a monolithic "Chat-with-PDF" baseline on a set of 50 financial queries requiring multi-hop reasoning (e.g., "Compare the operating margin of 2023 vs 2024").

| Method | Accuracy (%) | Hallucination Rate (%) | Avg Latency (s) |
| :--- | :---: | :---: | :---: |
| Baseline (Standard RAG) | 62.0% | 18.5% | **4.2s** |
| **Financial Swarm (Ours)** | **88.4%** | **4.2%** | 12.8s |

*Note: The Swarm architecture trades latency for significantly improved precision and reasoning depth.*

### 🎯 Performance Highlights
- **+26.4% accuracy improvement** over baseline
- **-14.3% reduction in hallucination rate**
- **Moderate latency increase** (3x) for 2x accuracy gain
- **Fully local execution** on consumer hardware

---

