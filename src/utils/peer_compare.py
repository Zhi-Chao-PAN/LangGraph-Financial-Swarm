"""
Peer comparison analysis module.
Automatically identifies industry peers and generates comparative analysis.
"""

from dataclasses import dataclass

from src.utils.calculator import FinancialCalculator, RatioResult
from src.utils.robustness import log_agent_action

# Industry peer groups (GICS-inspired)
PEER_GROUPS: dict[str, list[str]] = {
    "semiconductors": ["NVDA", "AMD", "INTC", "QCOM", "AVGO", "TXN"],
    "consumer_electronics": ["AAPL", "SONY", "SAMSUNG"],
    "software": ["MSFT", "ORCL", "CRM", "ADBE", "SAP"],
    "internet": ["GOOGL", "META", "SNAP", "PINS"],
    "ecommerce": ["AMZN", "BABA", "JD", "PDD", "EBAY"],
    "cloud": ["AMZN", "MSFT", "GOOGL", "CRM", "SNOW"],
    "ev": ["TSLA", "RIVN", "LCID", "NIO", "XPEV", "LI"],
    "fintech": ["V", "MA", "PYPL", "SQ", "COIN"],
    "social_media": ["META", "SNAP", "PINS", "RDDT"],
    "streaming": ["NFLX", "DIS", "WBD", "PARA"],
}

# Ticker to industry mapping
TICKER_INDUSTRY: dict[str, str] = {}
for _industry, _tickers in PEER_GROUPS.items():
    for _t in _tickers:
        TICKER_INDUSTRY.setdefault(_t, _industry)


@dataclass
class PeerMetrics:
    """Financial metrics for a single peer company."""

    ticker: str
    name: str
    ratios: list[RatioResult]

    def get_ratio(self, name: str) -> RatioResult | None:
        for r in self.ratios:
            if r.name == name:
                return r
        return None


@dataclass
class PeerComparison:
    """Result of a peer comparison analysis."""

    target_ticker: str
    industry: str
    peers: list[PeerMetrics]
    comparison_ratios: list[str]

    def to_markdown(self) -> str:
        """Generate a markdown comparison table."""
        if not self.peers:
            return f"No peer data available for {self.target_ticker}.\n"

        md = f"## Peer Comparison: {self.target_ticker} ({self.industry})\n\n"

        # Header
        tickers = [p.ticker for p in self.peers]
        md += "| Metric | " + " | ".join(tickers) + " |\n"
        md += "|:---|" + "|".join("---:" for _ in tickers) + "|\n"

        # Rows
        for ratio_name in self.comparison_ratios:
            vals = []
            for p in self.peers:
                r = p.get_ratio(ratio_name)
                vals.append(r.formatted() if r else "N/A")
            md += f"| {ratio_name} | " + " | ".join(vals) + " |\n"

        return md


def identify_peers(ticker: str, max_peers: int = 5) -> list[str]:
    """Identify peer companies for a given ticker."""
    ticker = ticker.upper()
    industry = TICKER_INDUSTRY.get(ticker)
    if not industry:
        return []

    peers = PEER_GROUPS.get(industry, [])
    # Put target first, then others
    result = [ticker] if ticker in peers else []
    for p in peers:
        if p != ticker and len(result) < max_peers:
            result.append(p)
    if ticker not in result:
        result.insert(0, ticker)
    return result[:max_peers]


def get_industry(ticker: str) -> str:
    """Get the industry for a ticker."""
    return TICKER_INDUSTRY.get(ticker.upper(), "unknown")


def compare_peers(
    target_ticker: str,
    peer_data: dict[str, dict],
    ratios: list[str] | None = None,
) -> PeerComparison:
    """
    Compare a target company against its peers.

    Args:
        target_ticker: The main company ticker
        peer_data: Dict of ticker -> {income: , balance: {}, cashflow: {}}
        ratios: List of ratio names to compare (default: key profitability + leverage)
    """
    if ratios is None:
        ratios = [
            "Gross Margin",
            "Operating Margin",
            "Net Margin",
            "ROE",
            "ROA",
            "Debt/Equity",
            "Current Ratio",
            "FCF Margin",
        ]

    industry = get_industry(target_ticker)
    peers = []

    for ticker, data in peer_data.items():
        calc = FinancialCalculator(
            income=data.get("income", {}),
            balance=data.get("balance", {}),
            cashflow=data.get("cashflow", {}),
        )
        all_ratios = calc.compute_all()
        peers.append(
            PeerMetrics(
                ticker=ticker,
                name=ticker,
                ratios=all_ratios,
            )
        )

    log_agent_action("PeerAnalysis", "Compare", f"{target_ticker} vs {len(peers) - 1} peers")

    return PeerComparison(
        target_ticker=target_ticker.upper(),
        industry=industry,
        peers=peers,
        comparison_ratios=ratios,
    )
