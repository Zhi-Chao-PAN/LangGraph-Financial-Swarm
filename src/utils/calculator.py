"""
Financial ratio calculator engine.
Computes 20+ financial ratios from income statement, balance sheet, and cash flow data.
"""

from dataclasses import dataclass


@dataclass
class RatioResult:
    """Result of a financial ratio calculation."""

    name: str
    value: float | None
    category: str
    description: str
    unit: str = "%"

    def formatted(self) -> str:
        if self.value is None:
            return "N/A"
        if self.unit == "%":
            return f"{self.value:.1f}%"
        if self.unit == "x":
            return f"{self.value:.2f}x"
        if self.unit == "$":
            return f"${self.value:,.0f}"
        return f"{self.value:.2f}"


def _safe_div(a: float | None, b: float | None) -> float | None:
    """Safe division, returns None if either operand is None or divisor is 0."""
    if a is None or b is None or b == 0:
        return None
    return a / b


def _pct(a: float | None, b: float | None) -> float | None:
    """Calculate percentage ratio."""
    result = _safe_div(a, b)
    return result * 100 if result is not None else None


class FinancialCalculator:
    """
    Calculates financial ratios from raw financial data.
    Accepts dicts with metric names as keys and numeric values.
    """

    def __init__(
        self,
        income: dict[str, float] | None = None,
        balance: dict[str, float] | None = None,
        cashflow: dict[str, float] | None = None,
    ):
        self.income = income or {}
        self.balance = balance or {}
        self.cashflow = cashflow or {}

    def _get(self, *keys: str) -> float | None:
        """Get a value from any of the financial statements."""
        for key in keys:
            for source in [self.income, self.balance, self.cashflow]:
                if key in source:
                    return source[key]
        return None

    # ─── Profitability Ratios ───

    def gross_margin(self) -> RatioResult:
        val = _pct(self._get("Gross Profit"), self._get("Total Revenue"))
        return RatioResult("Gross Margin", val, "profitability", "Gross profit as % of revenue")

    def operating_margin(self) -> RatioResult:
        val = _pct(self._get("Operating Income"), self._get("Total Revenue"))
        return RatioResult("Operating Margin", val, "profitability", "Operating income as % of revenue")

    def net_margin(self) -> RatioResult:
        val = _pct(self._get("Net Income"), self._get("Total Revenue"))
        return RatioResult("Net Margin", val, "profitability", "Net income as % of revenue")

    def roe(self) -> RatioResult:
        val = _pct(self._get("Net Income"), self._get("Stockholders Equity"))
        return RatioResult("ROE", val, "profitability", "Return on equity")

    def roa(self) -> RatioResult:
        val = _pct(self._get("Net Income"), self._get("Total Assets"))
        return RatioResult("ROA", val, "profitability", "Return on assets")

    def rd_ratio(self) -> RatioResult:
        val = _pct(self._get("Research And Development"), self._get("Total Revenue"))
        return RatioResult("R&D Ratio", val, "profitability", "R&D spending as % of revenue")

    # ─── Liquidity Ratios ───

    def current_ratio(self) -> RatioResult:
        val = _safe_div(
            self._get("Current Assets"),
            self._get("Current Liabilities"),
        )
        return RatioResult("Current Ratio", val, "liquidity", "Current assets / current liabilities", "x")

    def quick_ratio(self) -> RatioResult:
        current = self._get("Current Assets")
        inventory = self._get("Inventory") or 0
        current_liab = self._get("Current Liabilities")
        if current is not None and current_liab:
            val = (current - inventory) / current_liab
        else:
            val = None
        return RatioResult("Quick Ratio", val, "liquidity", "(Current assets - inventory) / current liabilities", "x")

    def cash_ratio(self) -> RatioResult:
        val = _safe_div(
            self._get("Cash And Cash Equivalents"),
            self._get("Current Liabilities"),
        )
        return RatioResult("Cash Ratio", val, "liquidity", "Cash / current liabilities", "x")

    # ─── Leverage Ratios ───

    def debt_to_equity(self) -> RatioResult:
        val = _safe_div(
            self._get("Total Debt"),
            self._get("Stockholders Equity"),
        )
        return RatioResult("Debt/Equity", val, "leverage", "Total debt / stockholders equity", "x")

    def debt_to_assets(self) -> RatioResult:
        val = _pct(
            self._get("Total Debt"),
            self._get("Total Assets"),
        )
        return RatioResult("Debt/Assets", val, "leverage", "Total debt as % of total assets")

    def equity_ratio(self) -> RatioResult:
        val = _pct(
            self._get("Stockholders Equity"),
            self._get("Total Assets"),
        )
        return RatioResult("Equity Ratio", val, "leverage", "Stockholders equity as % of total assets")

    # ─── Efficiency Ratios ───

    def asset_turnover(self) -> RatioResult:
        val = _safe_div(self._get("Total Revenue"), self._get("Total Assets"))
        return RatioResult("Asset Turnover", val, "efficiency", "Revenue / total assets", "x")

    def revenue_per_employee(self) -> RatioResult:
        val = _safe_div(self._get("Total Revenue"), self._get("Full Time Employees"))
        return RatioResult("Revenue/Employee", val, "efficiency", "Revenue per employee", "$")

    # ─── Cash Flow Ratios ───

    def fcf_margin(self) -> RatioResult:
        val = _pct(self._get("Free Cash Flow"), self._get("Total Revenue"))
        return RatioResult("FCF Margin", val, "cashflow", "Free cash flow as % of revenue")

    def ocf_ratio(self) -> RatioResult:
        val = _pct(self._get("Operating Cash Flow"), self._get("Total Revenue"))
        return RatioResult("OCF Ratio", val, "cashflow", "Operating cash flow as % of revenue")

    def capex_ratio(self) -> RatioResult:
        val = _pct(self._get("Capital Expenditure"), self._get("Total Revenue"))
        if val is not None:
            val = abs(val)  # CapEx is usually negative
        return RatioResult("CapEx Ratio", val, "cashflow", "Capital expenditure as % of revenue")

    # ─── All Ratios ───

    def compute_all(self) -> list[RatioResult]:
        """Compute all available financial ratios."""
        return [
            self.gross_margin(),
            self.operating_margin(),
            self.net_margin(),
            self.roe(),
            self.roa(),
            self.rd_ratio(),
            self.current_ratio(),
            self.quick_ratio(),
            self.cash_ratio(),
            self.debt_to_equity(),
            self.debt_to_assets(),
            self.equity_ratio(),
            self.asset_turnover(),
            self.revenue_per_employee(),
            self.fcf_margin(),
            self.ocf_ratio(),
            self.capex_ratio(),
        ]

    def compute_by_category(self, category: str) -> list[RatioResult]:
        """Compute ratios for a specific category."""
        return [r for r in self.compute_all() if r.category == category]

    def summary(self) -> dict[str, list[RatioResult]]:
        """Group all ratios by category."""
        result: dict[str, list[RatioResult]] = {}
        for r in self.compute_all():
            result.setdefault(r.category, []).append(r)
        return result
