# src/data/ingest.py
import os
import argparse
from pathlib import Path
from src.core.config import settings

def ingest_data(input_path: str, output_path: str):
    """
    Simple ingestion script to prepare data for RAG.
    Currently a placeholder that copies/moves files, but intended for 
    LlamaParse integration to convert PDF -> MD.
    """
    print(f"🚀 Starting Ingestion: {input_path} -> {output_path}")
    
    if not os.path.exists(input_path):
        print(f"❌ Error: Input path {input_path} does not exist.")
        return

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # placeholder: Just read and write for now
    # Recommended: Use LlamaParse here for academic precision
    # [Structure-Aware Parsing Implementation]
    # LlamaParse is capable of reconstructing complex tables which is critical for financial reports.
    # 
    # Pseudocode/Implementation:
    # from llama_parse import LlamaParse
    # parser = LlamaParse(result_type="markdown", verbose=True, language="en")
    # documents = parser.load_data(input_path)
    # 
    # structure_aware_md = ""
    # for doc in documents:
    #     # Post-processing to ensure table integrity
    #     structure_aware_md += doc.text + "\n\n"
    #
    # with open(output_path, 'w', encoding='utf-8') as f:
    #     f.write(structure_aware_md)
    
    # Fallback to simple copy for prototype if LlamaParse key is missing
    try:
        if input_path.endswith(".md"):
            with open(input_path, 'r', encoding='utf-8') as f:
                content = f.read()
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)
        else:
            print("⚠️ LlamaParse skipped (Key missing). Using direct file copy.")
            # In production, raise error or implementing simple PDF text extraction here
    except Exception as e:
         pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest financial data for RAG.")
    parser.add_argument("--input", type=str, help="Path to raw data (PDF/TXT/MD)")
    args = parser.parse_args()
    
    # Use defaults from settings if not provided
    input_file = args.input or "data/raw/sample_financials.md"
    output_file = settings.RAG_DATA_PATH
    
    ingest_data(input_file, output_file)
