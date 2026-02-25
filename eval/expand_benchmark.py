import json

# Load existing benchmark
with open("eval/benchmark.json", encoding="utf-8") as f:
    existing = json.load(f)

next_id = max(q["id"] for q in existing) + 1

new_questions = [
    # Multi-step reasoning
    {
        "question": "What was the difference between NVIDIA's and Apple's R&D spending in FY2025?",
        "answer": "NVIDIA spent $12,893M and Apple spent $31,370M on R&D in FY2025. Apple spent $18,477M more.",
        "type": "calculation",
        "difficulty": "hard"
    },
    {
        "question": "Which company had the highest revenue growth rate from FY2023 to FY2024: NVIDIA, Apple, or Microsoft?",
        "answer": "NVIDIA had the highest growth rate at approximately 126% (from $26,974M to $60,922M). Apple grew about 2% and Microsoft grew about 16%.",
        "type": "multi_hop",
        "difficulty": "hard"
    },
    {
        "question": "Calculate the combined total revenue of all three companies (NVIDIA, Apple, Microsoft) in FY2025.",
        "answer": "NVIDIA $130,497M + Apple $391,035M + Microsoft $245,122M = approximately $766,654M combined.",
        "type": "calculation",
        "difficulty": "hard"
    },
    {
        "question": "In FY2025, which company had the best operating margin: NVIDIA, Apple, or Microsoft?",
        "answer": "NVIDIA had the best operating margin at 65.2%, compared to Apple at 31.5% and Microsoft at 45.5%.",
        "type": "comparison",
        "difficulty": "hard"
    },
    {
        "question": "How did Microsoft's Intelligent Cloud segment revenue compare to NVIDIA's Data Center revenue in FY2025?",
        "answer": "Microsoft Intelligent Cloud revenue was approximately $105,360M and NVIDIA Data Center revenue was $115,199M in FY2025. NVIDIA's Data Center was higher.",
        "type": "comparison",
        "difficulty": "hard"
    },
    # Cross-year trend analysis
    {
        "question": "Did Apple's gross margin improve or decline from FY2021 to FY2025? By how much?",
        "answer": "Apple's gross margin improved from 41.8% in FY2021 to 46.2% in FY2025, an increase of 4.4 percentage points.",
        "type": "comparison",
        "difficulty": "medium"
    },
    {
        "question": "What was Microsoft's revenue CAGR from FY2021 to FY2025?",
        "answer": "Microsoft revenue grew from $168,088M (FY2021) to $245,122M (FY2025), a CAGR of approximately 9.9%.",
        "type": "calculation",
        "difficulty": "hard"
    },
    {
        "question": "Which company had the most consistent revenue growth from FY2021 to FY2025: NVIDIA, Apple, or Microsoft?",
        "answer": "Microsoft had the most consistent growth, increasing every year. NVIDIA had volatile growth (declined in FY2023 then surged). Apple had slight fluctuations.",
        "type": "multi_hop",
        "difficulty": "hard"
    },
    # Ratio and efficiency questions
    {
        "question": "Compare the R&D-to-revenue ratio of NVIDIA vs Microsoft in FY2025.",
        "answer": "NVIDIA R&D/Revenue = $12,893M/$130,497M = 9.9%. Microsoft R&D/Revenue = approximately 12-13%. Microsoft spent a higher percentage on R&D.",
        "type": "calculation",
        "difficulty": "hard"
    },
    {
        "question": "What was Apple's free cash flow per share in FY2025, given diluted shares of approximately 15.4 billion?",
        "answer": "Apple's operating cash flow was approximately $118,254M. Free cash flow approximately $108,807M. Per share: approximately $7.07.",
        "type": "calculation",
        "difficulty": "hard"
    },
    # Cross-company segment comparison
    {
        "question": "Which generated more revenue: Apple's iPhone segment or NVIDIA's entire company in FY2024?",
        "answer": "Apple iPhone revenue in FY2024 was approximately $201,183M. NVIDIA total revenue in FY2024 was $60,922M. iPhone alone generated over 3x NVIDIA's total.",
        "type": "comparison",
        "difficulty": "medium"
    },
    {
        "question": "How did NVIDIA's net income in FY2025 compare to Apple's net income in FY2025?",
        "answer": "NVIDIA net income FY2025 was approximately $72,880M. Apple net income FY2025 was approximately $93,736M. Apple's was higher by about $20,856M.",
        "type": "comparison",
        "difficulty": "medium"
    },
    # Geographic questions
    {
        "question": "Which company derived the highest percentage of revenue from the United States in FY2025?",
        "answer": "Microsoft derived approximately 50% from the US. NVIDIA derived about 34.7% ($45,274M/$130,497M). Apple derived about 42.7% from the Americas.",
        "type": "multi_hop",
        "difficulty": "hard"
    },
    # Balance sheet questions
    {
        "question": "Which company had the highest total assets in FY2025: NVIDIA, Apple, or Microsoft?",
        "answer": "Microsoft had the highest total assets at approximately $512,163M, compared to Apple at approximately $364,980M and NVIDIA at approximately $112,198M.",
        "type": "comparison",
        "difficulty": "medium"
    },
    {
        "question": "Calculate NVIDIA's debt-to-equity ratio in FY2025.",
        "answer": "NVIDIA total debt approximately $8,462M, total equity approximately $65,899M. Debt-to-equity ratio approximately 0.13.",
        "type": "calculation",
        "difficulty": "hard"
    },
    # Year-over-year change
    {
        "question": "How much did NVIDIA's operating cash flow increase from FY2024 to FY2025 in absolute terms?",
        "answer": "NVIDIA operating cash flow was $28,090M in FY2024 and $64,089M in FY2025, an increase of $35,999M.",
        "type": "calculation",
        "difficulty": "medium"
    },
    {
        "question": "Did Microsoft's operating margin improve or decline from FY2023 to FY2025?",
        "answer": "Microsoft operating margin improved from approximately 41.2% in FY2023 to approximately 45.5% in FY2025.",
        "type": "comparison",
        "difficulty": "medium"
    },
    # Multi-company multi-metric
    {
        "question": "Rank NVIDIA, Apple, and Microsoft by net income in FY2025 from highest to lowest.",
        "answer": "1. Apple: ~$93,736M, 2. NVIDIA: ~$72,880M, 3. Microsoft: ~$90,510M. Order: Apple > Microsoft > NVIDIA.",
        "type": "multi_hop",
        "difficulty": "hard"
    },
    {
        "question": "Which company had the lowest gross margin in FY2023: NVIDIA, Apple, or Microsoft?",
        "answer": "Apple had approximately 44.1%, Microsoft had approximately 68.9%, NVIDIA had 56.9%. Apple had the lowest gross margin in FY2023.",
        "type": "comparison",
        "difficulty": "medium"
    },
    {
        "question": "How many years between FY2021 and FY2025 did NVIDIA's revenue exceed $50 billion?",
        "answer": "NVIDIA revenue: FY2021 $16,675M, FY2022 $26,914M, FY2023 $26,974M, FY2024 $60,922M, FY2025 $130,497M. Two years (FY2024 and FY2025) exceeded $50B.",
        "type": "multi_hop",
        "difficulty": "hard"
    },
]

# Add IDs
for i, q in enumerate(new_questions):
    q["id"] = next_id + i

# Combine
combined = existing + new_questions
with open("eval/benchmark.json", "w", encoding="utf-8") as f:
    json.dump(combined, f, indent=2, ensure_ascii=False)

print(f"Benchmark expanded: {len(existing)} -> {len(combined)} questions")
print(f"New questions: {len(new_questions)}")
for q in new_questions:
    print(f"  Q{q['id']}: [{q['difficulty']}/{q['type']}] {q['question'][:60]}...")
