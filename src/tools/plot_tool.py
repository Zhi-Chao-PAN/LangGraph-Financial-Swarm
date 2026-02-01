# src/tools/plot_tool.py
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import io
import os
from langchain_core.tools import tool
from src.utils.robustness import log_agent_action
from src.core.config import settings
from src.utils.validation import validate_dataframe

# Set academic style
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12, 'figure.dpi': 300})

@tool
@validate_dataframe
def create_plot(data_str: str, plot_type: str, title: str, xlabel: str, ylabel: str) -> str:
    """
    Generate a publication-quality plot and save the raw data for verification.

    Args:
        data_str (str): JSON or CSV string containing the data.
        plot_type (str): Type of chart ('bar', 'line', 'pie').
        title (str): Title of the chart.
        xlabel (str): Label for the X-axis.
        ylabel (str): Label for the Y-axis.

    Returns:
        str: Success message with file paths.
    """
    try:
        log_agent_action("Quant", "GenerateChart", f"Type: {plot_type}, Title: {title}")
        
        # Robust parsing
        try:
             df = pd.read_csv(io.StringIO(data_str))
        except:
             import json
             data = json.loads(data_str)
             df = pd.DataFrame(data)

        # 1. Save Raw Data (Audit Trail)
        output_dir = settings.OUTPUT_DIR
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        safe_title = "".join([c if c.isalnum() else "_" for c in title]).lower()
        csv_path = os.path.join(output_dir, f"{safe_title}.csv")
        df.to_csv(csv_path, index=False)

        # 2. Generate Plot
        plt.figure(figsize=(10, 6))
        
        if plot_type == "bar":
            ax = sns.barplot(x=df.columns[0], y=df.columns[1], data=df, palette="viridis")
            if plot_type == 'bar':
                sns.barplot(data=df, x='label', y='value', hue='label', palette='viridis', legend=False)
        elif plot_type == 'line':
            sns.lineplot(data=df, x='year', y='value', marker='o')
        elif plot_type == 'scatter':
            sns.scatterplot(data=df, x='year', y='value', s=100)
        elif plot_type == 'hist':
            sns.histplot(data=df, x='value', kde=True)
        else:
            sns.barplot(data=df, x='label', y='value') # Default
        
        plt.title(title, fontsize=16, fontweight='bold', pad=20)
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.tight_layout()
        
        png_path = os.path.join(output_dir, f"{safe_title}.png")
        plt.savefig(png_path, bbox_inches='tight')
        plt.close()
        
        return f"Chart generated: {png_path} (Raw Data: {csv_path})"
    except Exception as e:
        error_msg = f"Error creating plot: {str(e)}"
        log_agent_action("Quant", "Error", error_msg)
        return error_msg
