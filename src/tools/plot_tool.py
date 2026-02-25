import io
import json
import os
import uuid

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from langchain_core.tools import tool

from src.core.config import settings
from src.utils.robustness import log_agent_action
from src.utils.validation import validate_dataframe

COLORS = ["#2ecc71", "#3498db", "#e74c3c", "#f39c12", "#9b59b6", "#1abc9c", "#e67e22", "#34495e"]


@tool
@validate_dataframe
def create_plot(data_str: str, plot_type: str, title: str, xlabel: str, ylabel: str) -> str:
    """
    Generate a publication-quality plot and save the raw data for verification.

    Args:
        data_str: JSON string containing the data (list of dicts).
        plot_type: Type of chart ('bar', 'line', 'scatter', 'hist', 'pie', 'stacked_bar').
        title: Title of the chart.
        xlabel: Label for the X-axis.
        ylabel: Label for the Y-axis.

    Returns:
        Success message with file paths.
    """
    try:
        log_agent_action("Quant", "GenerateChart", f"Type: {plot_type}, Title: {title}")

        MAX_DATA_SIZE = 1024 * 50
        if len(data_str) > MAX_DATA_SIZE:
            return "Error: Data size exceeds limit (50KB)."

        try:
            df = pd.read_csv(io.StringIO(data_str))
        except Exception:
            data = json.loads(data_str)
            df = pd.DataFrame(data)

        output_dir = settings.OUTPUT_DIR
        os.makedirs(output_dir, exist_ok=True)

        safe_title = "".join([c if c.isalnum() else "_" for c in title]).lower()
        uid = uuid.uuid4().hex[:8]
        csv_path = os.path.join(output_dir, f"{safe_title}_{uid}.csv")
        df.to_csv(csv_path, index=False)

        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

        col_x = df.columns[0]
        col_y = df.columns[1] if len(df.columns) > 1 else df.columns[0]

        if plot_type == "pie":
            values = df[col_y].astype(float)
            labels = df[col_x].astype(str)
            ax.pie(values, labels=labels, autopct="%1.1f%%",
                   colors=COLORS[:len(values)], startangle=90)
            ax.set_ylabel("")

        elif plot_type == "stacked_bar":
            categories = df[col_x].astype(str).tolist()
            value_cols = df.columns[1:]
            bottom = [0.0] * len(df)
            x_pos = range(len(categories))
            for i, vc in enumerate(value_cols):
                vals = df[vc].astype(float).tolist()
                ax.bar(x_pos, vals, bottom=bottom, label=vc,
                       color=COLORS[i % len(COLORS)])
                bottom = [b + v for b, v in zip(bottom, vals, strict=False)]
            ax.set_xticks(list(x_pos))
            ax.set_xticklabels(categories, rotation=45, ha="right")
            ax.legend(loc="upper left", fontsize=9)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)

        elif plot_type == "bar":
            if len(df.columns) >= 3:
                categories = df[col_x].astype(str).tolist()
                value_cols = df.columns[1:]
                x_pos = list(range(len(categories)))
                width = 0.8 / len(value_cols)
                for i, vc in enumerate(value_cols):
                    offset = (i - len(value_cols) / 2 + 0.5) * width
                    ax.bar([p + offset for p in x_pos],
                           df[vc].astype(float).tolist(),
                           width=width, label=vc,
                           color=COLORS[i % len(COLORS)])
                ax.set_xticks(x_pos)
                ax.set_xticklabels(categories, rotation=45, ha="right")
                ax.legend(fontsize=9)
            else:
                x_vals = df[col_x].astype(str).tolist()
                y_vals = df[col_y].astype(float).tolist()
                ax.bar(x_vals, y_vals, color=COLORS[0])
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)

        elif plot_type == "line":
            x_vals = df[col_x].astype(str).tolist()
            if len(df.columns) >= 3:
                for i, vc in enumerate(df.columns[1:]):
                    ax.plot(x_vals, df[vc].astype(float).tolist(),
                            marker="o", label=vc, color=COLORS[i % len(COLORS)])
                ax.legend(fontsize=9)
            else:
                ax.plot(x_vals, df[col_y].astype(float).tolist(),
                        marker="o", color=COLORS[0])
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)

        elif plot_type == "scatter":
            ax.scatter(df[col_x].astype(float), df[col_y].astype(float),
                       s=100, color=COLORS[0])
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)

        elif plot_type == "hist":
            ax.hist(df[col_y].astype(float), bins=20, color=COLORS[0], edgecolor="white")
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)

        else:
            x_vals = df[col_x].astype(str).tolist()
            y_vals = df[col_y].astype(float).tolist()
            ax.bar(x_vals, y_vals, color=COLORS[0])
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)

        ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
        ax.grid(True, alpha=0.3)

        png_path = os.path.join(output_dir, f"{safe_title}_{uid}.png")
        fig.savefig(png_path, dpi=150, pad_inches=0.3)
        plt.close(fig)

        return f"Chart generated: {png_path} (Raw Data: {csv_path})"
    except Exception as e:
        error_msg = f"Error creating plot: {e}"
        log_agent_action("Quant", "Error", error_msg)
        plt.close("all")
        return error_msg
