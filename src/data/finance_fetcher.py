"""
Yahoo Finance data fetcher for LangGraph Financial Swarm.
Fetches real-time financial data for any publicly traded company
and converts it to structured Markdown for RAG ingestion.

Usage:
    python -m src.data.finance_fetcher --ticker AAPL NVDA GOOGL
    python -m src.data.finance_fetcher --ticker 600519.SS  # A-shares
"""

import argparse
import json
import sqlite3
import time
from pathlib import Path

import yfinance as yf

from src.core.config import settings
from src.utils.robustness import log_agent_action

# Cache database path
CACHE_DB = settings.DATA_DIR / "finance_cache.db"


def _init_cache():
    """Initialize SQLite cache database."""
    CACHE_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(CACHE_DB))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS finance_cache (
            ticker TEXT PRIMARY KEY,
            data TEXT,
            markdown TEXT,
            fetched_at REAL,
            ttl INTEGER DEFAULT 86400
        )
    """)
    conn.commit()
    return conn


def _get_cached(conn: sqlite3.Connection, ticker: str, ttl: int = 86400) -> str | None:
    """Get cached markdown if still valid."""
    row = conn.execute(
        "SELECT markdown, fetched_at FROM finance_cache WHERE ticker = ?",
        (ticker.upper(),),
    ).fetchone()
    if row and (time.time() - row[1]) < ttl:
        return row[0]
    return None


def _set_cache(conn: sqlite3.Connection, ticker: str, data: dict, markdown: str):
    """Store fetched data in cache."""
    conn.execute(
        "INSERT OR REPLACE INTO finance_cache (ticker, data, markdown, fetched_at) VALUES (?, ?, ?, ?)",
        (ticker.upper(), json.dumps(data, default=str), markdown, time.time()),
    )
    conn.commit()


def fetch_company_data(ticker: str) -> dict:
    """
    Fetch financial data for a company from Yahoo Finance.
    Returns a dict with company info, income statement, balance sheet, etc.
    """
    log_agent_action("Fetcher", "Fetch", f"Fetching data for {ticker}")
    stock = yf.Ticker(ticker)

    info = stock.info or {}
    company_name = info.get("longName") or info.get("shortName") or ticker.upper()

    # Income statement (annual)
    income_stmt = stock.income_stmt
    # Balance sheet (annual)
    balance_sheet = stock.balance_sheet
    # Cash flow (annual)
    cashflow = stock.cashflow

    return {
        "ticker": ticker.upper(),
        "name": company_name,
        "info": {
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "country": info.get("country", "N/A"),
            "currency": info.get("currency", "USD"),
            "market_cap": info.get("marketCap"),
            "website": info.get("website", "N/A"),
        },
        "income_stmt": _df_to_dict(income_stmt) if income_stmt is not None else {},
        "balance_sheet": _df_to_dict(balance_sheet) if balance_sheet is not None else {},
        "cashflow": _df_to_dict(cashflow) if cashflow is not None else {},
    }


def _df_to_dict(df) -> dict:
    """Convert a pandas DataFrame to a serializable dict."""
    if df is None or df.empty:
        return {}
    result = {}
    for col in df.columns:
        year = str(col.year) if hasattr(col, "year") else str(col)
        result[year] = {}
        for idx in df.index:
            val = df.loc[idx, col]
            if val is not None and str(val) != "nan":
                result[year][str(idx)] = float(val)
    return result


def _format_number(val: float, currency: str = "USD") -> str:
    """Format a number for display in millions."""
    if abs(val) >= 1e9:
        return f"{val / 1e6:,.0f}"
    if abs(val) >= 1e6:
        return f"{val / 1e6:,.1f}"
    return f"{val:,.0f}"


def _build_table(data: dict, metrics: list[str], currency: str = "USD") -> str:
    """Build a markdown table from financial data."""
    if not data:
        return "*No data available.*\n"

    years = sorted(data.keys(), reverse=False)[-5:]  # Last 5 years
    if not years:
        return "*No data available.*\n"

    # Header
    header = f"| Metric ({currency} Millions) | " + " | ".join(f"FY{y}" for y in years) + " |"
    separator = "|:---|" + "|".join("---:" for _ in years) + "|"

    rows = []
    for metric in metrics:
        vals = []
        found = False
        for y in years:
            v = data.get(y, {}).get(metric)
            if v is not None:
                vals.append(_format_number(v))
                found = True
            else:
                vals.append("N/A")
        if found:
            display_name = metric.replace("And", " & ").replace("  ", " ")
            rows.append(f"| {display_name} | " + " | ".join(vals) + " |")

    if not rows:
        return "*No data available.*\n"

    return header + "\n" + separator + "\n" + "\n".join(rows) + "\n"


def data_to_markdown(data: dict) -> str:
    """Convert fetched financial data to structured Markdown."""
    info = data["info"]
    md = f"# {data['name']} — Financial Summary\n\n"
    md += f"Source: Yahoo Finance ({data['ticker']}). Data retrieved automatically.\n\n"
    md += f"Sector: {info['sector']} | Industry: {info['industry']} | Country: {info['country']}\n\n"

    if info.get("market_cap"):
        mc = info["market_cap"]
        if mc >= 1e12:
            md += f"Market Cap: ${mc / 1e12:.2f}T\n\n"
        elif mc >= 1e9:
            md += f"Market Cap: ${mc / 1e9:.1f}B\n\n"

    md += "---\n\n"

    # Income Statement
    income_metrics = [
        "Total Revenue",
        "Cost Of Revenue",
        "Gross Profit",
        "Research And Development",
        "Selling General And Administration",
        "Operating Income",
        "Net Income",
        "Basic EPS",
        "Diluted EPS",
    ]
    md += "## Consolidated Income Statement\n\n"
    md += _build_table(data["income_stmt"], income_metrics, info["currency"])
    md += "\n"

    # Key ratios (calculated)
    md += "## Key Financial Ratios\n\n"
    md += _build_ratios_table(data["income_stmt"])
    md += "\n"

    # Balance Sheet
    balance_metrics = [
        "Total Assets",
        "Total Liabilities Net Minority Interest",
        "Stockholders Equity",
        "Cash And Cash Equivalents",
        "Total Debt",
        "Net Debt",
    ]
    md += "## Balance Sheet Highlights\n\n"
    md += _build_table(data["balance_sheet"], balance_metrics, info["currency"])
    md += "\n"

    # Cash Flow
    cashflow_metrics = [
        "Operating Cash Flow",
        "Capital Expenditure",
        "Free Cash Flow",
        "Repurchase Of Capital Stock",
    ]
    md += "## Cash Flow Statement\n\n"
    md += _build_table(data["cashflow"], cashflow_metrics, info["currency"])

    md += f"\n*Data compiled from Yahoo Finance. Ticker: {data['ticker']}.*\n"
    return md


def _build_ratios_table(income_data: dict) -> str:
    """Calculate and build a ratios table from income statement data."""
    if not income_data:
        return "*No data available.*\n"

    years = sorted(income_data.keys(), reverse=False)[-5:]
    if not years:
        return "*No data available.*\n"

    header = "| Ratio | " + " | ".join(f"FY{y}" for y in years) + " |"
    separator = "|:---|" + "|".join("---:" for _ in years) + "|"

    ratios = {
        "Gross Margin (%)": ("Gross Profit", "Total Revenue"),
        "Operating Margin (%)": ("Operating Income", "Total Revenue"),
        "Net Margin (%)": ("Net Income", "Total Revenue"),
        "R&D as % of Revenue": ("Research And Development", "Total Revenue"),
    }

    rows = []
    for name, (numerator, denominator) in ratios.items():
        vals = []
        found = False
        for y in years:
            n = income_data.get(y, {}).get(numerator)
            d = income_data.get(y, {}).get(denominator)
            if n is not None and d is not None and d != 0:
                vals.append(f"{n / d * 100:.1f}")
                found = True
            else:
                vals.append("N/A")
        if found:
            rows.append(f"| {name} | " + " | ".join(vals) + " |")

    # YoY Revenue Growth
    rev_vals = []
    prev_rev = None
    found_growth = False
    for y in years:
        rev = income_data.get(y, {}).get("Total Revenue")
        if rev is not None and prev_rev is not None and prev_rev != 0:
            growth = (rev - prev_rev) / abs(prev_rev) * 100
            rev_vals.append(f"{growth:.1f}")
            found_growth = True
        else:
            rev_vals.append("N/A")
        prev_rev = rev
    if found_growth:
        rows.append("| YoY Revenue Growth (%) | " + " | ".join(rev_vals) + " |")

    if not rows:
        return "*No data available.*\n"

    return header + "\n" + separator + "\n" + "\n".join(rows) + "\n"


def fetch_and_convert(ticker: str, cache_ttl: int = 86400, force: bool = False) -> str:
    """
    Fetch company data and convert to Markdown.
    Uses cache if available and not expired.
    """
    conn = _init_cache()
    try:
        if not force:
            cached = _get_cached(conn, ticker, ttl=cache_ttl)
            if cached:
                log_agent_action("Fetcher", "Cache Hit", f"{ticker} (cached)")
                return cached

        data = fetch_company_data(ticker)
        markdown = data_to_markdown(data)
        _set_cache(conn, ticker, data, markdown)
        log_agent_action("Fetcher", "Done", f"{ticker}: {len(markdown)} chars")
        return markdown
    finally:
        conn.close()


def fetch_multiple(tickers: list[str], output_path: str | None = None, cache_ttl: int = 86400) -> str:
    """
    Fetch multiple companies and combine into a single Markdown document.
    Optionally write to output_path.
    """
    sections = []
    for ticker in tickers:
        try:
            md = fetch_and_convert(ticker, cache_ttl=cache_ttl)
            sections.append(md)
        except Exception as e:
            log_agent_action("Fetcher", "Error", f"{ticker}: {e}", level="error")
            sections.append(f"# {ticker} — Error\n\nFailed to fetch data: {e}\n")

    combined = "\n\n---\n\n".join(sections)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(combined, encoding="utf-8")
        log_agent_action("Fetcher", "Save", f"Written to {output_path}")

    return combined


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Fetch financial data from Yahoo Finance")
    parser.add_argument("--ticker", "-t", nargs="+", required=True, help="Stock ticker(s)")
    parser.add_argument("--output", "-o", default=None, help="Output markdown file path")
    parser.add_argument("--ttl", type=int, default=86400, help="Cache TTL in seconds (default: 24h)")
    parser.add_argument("--force", action="store_true", help="Force refresh (ignore cache)")
    args = parser.parse_args()

    if args.force:
        # Clear cache for these tickers
        conn = _init_cache()
        for t in args.ticker:
            conn.execute("DELETE FROM finance_cache WHERE ticker = ?", (t.upper(),))
        conn.commit()
        conn.close()

    result = fetch_multiple(args.ticker, output_path=args.output, cache_ttl=args.ttl)
    print(result)


if __name__ == "__main__":
    main()
