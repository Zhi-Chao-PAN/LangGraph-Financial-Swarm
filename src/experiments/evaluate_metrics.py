
import json
import re
import argparse
from typing import List, Dict, Any

def normalize_number(s: str) -> float:
    """
    Extracts the first floating point number from a string, normalizing units if possible.
    Very basic implementation for exact match.
    """
    # Remove commas
    s = s.replace(",", "")
    # Find float pattern
    match = re.search(r"[-+]?\d*\.\d+|\d+", s)
    if match:
        return float(match.group())
    return None

def check_exact_match(model_answer: str, ground_truth: str) -> bool:
    """
    Checks if the model answer contains the ground truth number.
    Allows for some formatting differences (commas, etc).
    """
    if not ground_truth or ground_truth == "N/A":
        return False
        
    gt_val = normalize_number(str(ground_truth))
    if gt_val is None:
        return False
        
    # Heuristic: Check if the number appears in the answer
    # We normalize numbers in the model answer and see if any match the GT within a tolerance
    
    # Extract all numbers from model answer
    # This is a 'loose' exact match - if the number is present, we count it.
    # A stricter version would parse the answer structurally.
    
    # Remove commas from model answer for regex
    answer_clean = model_answer.replace(",", "")
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", answer_clean)
    
    for num_str in numbers:
        try:
            val = float(num_str)
            if abs(val - gt_val) < 0.01: # Check within tolerance
                return True
        except:
            continue
            
    return False

def calculate_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(results)
    if total == 0:
        return {"accuracy": 0, "avg_latency": 0}
        
    correct_count = 0
    total_latency = 0
    
    for row in results:
        is_correct = check_exact_match(row.get("model_answer", ""), row.get("ground_truth", ""))
        row["is_correct"] = is_correct # Annotate result
        if is_correct:
            correct_count += 1
        total_latency += row.get("latency_s", 0)
        
    return {
        "accuracy": correct_count / total,
        "avg_latency": total_latency / total,
        "total_samples": total
    }

def main():
    parser = argparse.ArgumentParser(description="Calculate metrics for RAG experiment")
    parser.add_argument("input_file", help="Path to JSON results file")
    args = parser.parse_args()
    
    try:
        with open(args.input_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        metrics = calculate_metrics(data)
        
        print("\n=== Evaluation Metrics ===")
        print(f"Input File: {args.input_file}")
        print(f"Total Samples: {metrics['total_samples']}")
        print(f"Exact Match Accuracy: {metrics['accuracy']:.2%}")
        print(f"Average Latency: {metrics['avg_latency']:.4f}s")
        print("==========================\n")
        
        # Optionally save annotated results
        output_file = args.input_file.replace(".json", "_scored.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"Scored results saved to: {output_file}")
        
    except Exception as e:
        print(f"Error calculating metrics: {e}")

if __name__ == "__main__":
    main()
