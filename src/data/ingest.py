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
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
            
        print(f"✅ Ingestion Complete. Data saved to {output_path}")
    except Exception as e:
        print(f"❌ Ingestion Failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest financial data for RAG.")
    parser.add_argument("--input", type=str, help="Path to raw data (PDF/TXT/MD)")
    args = parser.parse_args()
    
    # Use defaults from settings if not provided
    input_file = args.input or "data/raw/sample_financials.md"
    output_file = settings.RAG_DATA_PATH
    
    ingest_data(input_file, output_file)
