"""
Financial report generator.
Generates structured financial health reports in Markdown format.
Supports Chinese and English output.
"""

from dataclasses import dataclass

from src.utils.calculator import FinancialCalculator
from src.utils.trend import analyze_multiple_trends


@dataclass
class ReportConfig:
    """Configuration for report generation."""

    language: str = "en"  # "en" or "zh"
    include_ratios: bool = True
    include_trends: bool = True
    include_risk: bool = True
    include_summary: bool = True


# Bilingual labels
LABELS = {
    "en": {
        "title": "Financial Health Report",
        "overview": "Company Overview",
        "profitability": "Profitability Analysis",
        "liquidity": "Liquidity & Solvency",
        "leverage": "Leverage Analysis",
        "cashflow": "Cash Flow Analysis",
        "trends": "Trend Analysis",
        "risk": "Risk Indicators",
        "summary": "Executive Summary",
        "strong": "Strong",
        "moderate": "Moderate",
        "weak": "Weak",
        "up": "Increasing",
        "down": "Decreasing",
        "flat": "Stable",
        "no_data": "No data available",
        "gross_margin": "Gross Margin",
        "op_margin": "Operating Margin",
        "net_margin": "Net Margin",
        "health_good": "The company shows strong financial health with solid profitability and manageable leverage.",
        "health_moderate": "The company shows moderate financial health. Some metrics warrant monitoring.",
        "health_weak": "The company shows signs of financial stress. Key metrics are below healthy thresholds.",
    },
    "zh": {
        "title": "财务健康报告",
        "overview": "公司概览",
        "profitability": "盈利能力分析",
        "liquidity": "流动性与偿债能力",
        "leverage": "杠杆分析",
        "cashflow": "现金流分析",
        "trends": "趋势分析",
        "risk": "风险指标",
        "summary": "总结摘要",
        "strong": "强",
        "moderate": "中等",
        "weak": "弱",
        "up": "上升",
        "down": "下降",
        "flat": "稳定",
        "no_data": "暂无数据",
        "gross_margin": "毛利率",
        "op_margin": "营业利润率",
        "net_margin": "净利率",
        "health_good": "公司财务状况良好，盈利能力强劲，杠杆水平可控。",
        "health_moderate": "公司财务状况中等，部分指标需要关注。",
        "health_weak": "公司存在财务压力迹象，关键指标低于健康阈值。",
    },
}


def _rate(value: float | None, good: float, bad: float, lang: str = "en") -> str:
    """Rate a metric value as strong/moderate/weak."""
    L = LABELS[lang]
    if value is None:
        return L["no_data"]
    if good > bad:  # Higher is better (e.g., margin)
        if value >= good:
            return L["strong"]
        if value >= bad:
            return L["moderate"]
        return L["weak"]
    # Lower is better (e.g., debt ratio)
    if value <= good:
        return L["strong"]
    if value <= bad:
        return L["moderate"]
    return L["weak"]


def generate_report(
    company_name: str,
    ticker: str,
    income: dict[str, float] | None = None,
    balance: dict[str, float] | None = None,
    cashflow: dict[str, float] | None = None,
    historical: dict[str, dict[str, float]] | None = None,
    config: ReportConfig | None = None,
) -> str:
    """
    Generate a financial health report in Markdown.

    Args:
        company_name: Company display name
        ticker: Stock ticker
        income: Latest income statement data
        balance: Latest balance sheet data
        cashflow: Latest cash flow data
        historical: Multi-year data for trend analysis {year: {metric: value}}
        config: Report configuration
    """
    if config is None:
        config = ReportConfig()

    L = LABELS.get(config.language, LABELS["en"])
    calc = FinancialCalculator(income=income, balance=balance, cashflow=cashflow)

    md = f"# {company_name} ({ticker}) — {L['title']}\n\n"

    # ─── Ratios Section ───
    if config.include_ratios:
        # Profitability
        md += f"## {L['profitability']}\n\n"
        md += "| Metric | Value | Rating |\n|:---|---:|:---|\n"
        for r in calc.compute_by_category("profitability"):
            rating = _rate(r.value, 30, 10, config.language) if "Margin" in r.name else ""
            md += f"| {r.name} | {r.formatted()} | {rating} |\n"
        md += "\n"

        # Liquidity
        md += f"## {L['liquidity']}\n\n"
        md += "| Metric | Value | Rating |\n|:---|---:|:---|\n"
        for r in calc.compute_by_category("liquidity"):
            rating = _rate(r.value, 1.5, 1.0, config.language) if r.value else ""
            md += f"| {r.name} | {r.formatted()} | {rating} |\n"
        md += "\n"

        # Leverage
        md += f"## {L['leverage']}\n\n"
        md += "| Metric | Value | Rating |\n|:---|---:|:---|\n"
        for r in calc.compute_by_category("leverage"):
            md += f"| {r.name} | {r.formatted()} |\n"
        md += "\n"

        # Cash Flow
        md += f"## {L['cashflow']}\n\n"
        md += "| Metric | Value |\n|:---|---:|\n"
        for r in calc.compute_by_category("cashflow"):
            md += f"| {r.name} | {r.formatted()} |\n"
        md += "\n"

    # ─── Trends Section ───
    if config.include_trends and historical:
        md += f"## {L['trends']}\n\n"
        metrics = ["Total Revenue", "Net Income", "Gross Profit", "Operating Income"]
        trends = analyze_multiple_trends(historical, metrics)
        for t in trends:
            arrow = {"up": "↑", "down": "↓", "flat": "→"}.get(t.direction, "?")
            direction_label = L.get(t.direction, t.direction)
            sig = f"R²={t.r_squared:.2f}"
            md += f"- {t.metric}: {arrow} {direction_label} ({sig})"
            if t.forecast_next is not None:
                md += f" — forecast: {t.forecast_next:,.0f}"
            md += "\n"
        md += "\n"

    # ─── Risk Section ───
    if config.include_risk:
        md += f"## {L['risk']}\n\n"
        risks = []
        gm = calc.gross_margin()
        if gm.value is not None and gm.value < 20:
            risks.append(f"Low gross margin ({gm.formatted()})")
        de = calc.debt_to_equity()
        if de.value is not None and de.value > 2.0:
            risks.append(f"High leverage (D/E = {de.formatted()})")
        cr = calc.current_ratio()
        if cr.value is not None and cr.value < 1.0:
            risks.append(f"Low liquidity (Current Ratio = {cr.formatted()})")
        nm = calc.net_margin()
        if nm.value is not None and nm.value < 0:
            risks.append(f"Negative net margin ({nm.formatted()})")

        if risks:
            for risk in risks:
                md += f"- ⚠️ {risk}\n"
        else:
            md += "- ✅ No significant risk indicators detected.\n"
        md += "\n"

    # ─── Summary ───
    if config.include_summary:
        md += f"## {L['summary']}\n\n"
        score = 0
        total = 0
        for r in calc.compute_all():
            if r.value is not None:
                total += 1
                if (
                    (r.category == "profitability" and r.value > 15)
                    or (r.category == "liquidity" and r.value > 1.0)
                    or (r.category == "leverage" and r.name == "Debt/Equity" and r.value < 1.5)
                    or (r.category == "cashflow" and r.value > 10)
                ):
                    score += 1

        if total == 0:
            md += L["no_data"] + "\n"
        elif score / total >= 0.6:
            md += L["health_good"] + "\n"
        elif score / total >= 0.3:
            md += L["health_moderate"] + "\n"
        else:
            md += L["health_weak"] + "\n"

    return md
