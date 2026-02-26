"""
Portfolio tracker for personal investment management.
Track holdings, calculate returns, and generate portfolio reports.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Holding:
    """A single stock holding."""

    ticker: str
    shares: float
    avg_cost: float  # Average cost per share
    added_at: float = 0.0

    @property
    def total_cost(self) -> float:
        return self.shares * self.avg_cost

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "shares": self.shares,
            "avg_cost": self.avg_cost,
            "total_cost": round(self.total_cost, 2),
            "added_at": self.added_at,
        }


@dataclass
class PortfolioSummary:
    """Summary of portfolio performance."""

    total_cost: float = 0.0
    total_value: float = 0.0
    total_return: float = 0.0
    return_pct: float = 0.0
    holdings_count: int = 0
    top_holding: str = ""
    top_weight: float = 0.0

    def to_dict(self) -> dict:
        return {
            "total_cost": round(self.total_cost, 2),
            "total_value": round(self.total_value, 2),
            "total_return": round(self.total_return, 2),
            "return_pct": round(self.return_pct, 2),
            "holdings_count": self.holdings_count,
            "top_holding": self.top_holding,
            "top_weight_pct": round(self.top_weight, 2),
        }

    def to_markdown(self, language: str = "en") -> str:
        if language == "zh":
            return (
                f"## 📊 投资组合概览\n"
                f"- 持仓数量: {self.holdings_count}\n"
                f"- 总成本: ${self.total_cost:,.2f}\n"
                f"- 当前市值: ${self.total_value:,.2f}\n"
                f"- 总收益: ${self.total_return:,.2f} ({self.return_pct:+.1f}%)\n"
                f"- 最大持仓: {self.top_holding} ({self.top_weight:.1f}%)\n"
            )
        return (
            f"## 📊 Portfolio Summary\n"
            f"- Holdings: {self.holdings_count}\n"
            f"- Total Cost: ${self.total_cost:,.2f}\n"
            f"- Current Value: ${self.total_value:,.2f}\n"
            f"- Total Return: ${self.total_return:,.2f} ({self.return_pct:+.1f}%)\n"
            f"- Top Holding: {self.top_holding} ({self.top_weight:.1f}%)\n"
        )


class Portfolio:
    """Personal investment portfolio tracker."""

    def __init__(self, name: str = "My Portfolio") -> None:
        self.name = name
        self._holdings: dict[str, Holding] = {}
        self._created_at = time.time()

    def add(self, ticker: str, shares: float, cost_per_share: float) -> Holding:
        """Add shares to portfolio. Averages cost if already held."""
        ticker = ticker.upper()
        if ticker in self._holdings:
            existing = self._holdings[ticker]
            total_shares = existing.shares + shares
            total_cost = existing.total_cost + (shares * cost_per_share)
            existing.avg_cost = total_cost / total_shares if total_shares > 0 else 0
            existing.shares = total_shares
            return existing

        holding = Holding(
            ticker=ticker,
            shares=shares,
            avg_cost=cost_per_share,
            added_at=time.time(),
        )
        self._holdings[ticker] = holding
        return holding

    def remove(self, ticker: str, shares: float | None = None) -> bool:
        """Remove shares or entire position."""
        ticker = ticker.upper()
        if ticker not in self._holdings:
            return False
        if shares is None or shares >= self._holdings[ticker].shares:
            del self._holdings[ticker]
        else:
            self._holdings[ticker].shares -= shares
        return True

    def get(self, ticker: str) -> Holding | None:
        return self._holdings.get(ticker.upper())

    @property
    def holdings(self) -> list[Holding]:
        return list(self._holdings.values())

    @property
    def tickers(self) -> list[str]:
        return list(self._holdings.keys())

    def summary(self, current_prices: dict[str, float] | None = None) -> PortfolioSummary:
        """Calculate portfolio summary with optional current prices."""
        if not self._holdings:
            return PortfolioSummary()

        total_cost = sum(h.total_cost for h in self._holdings.values())

        if current_prices:
            total_value = sum(h.shares * current_prices.get(h.ticker, h.avg_cost) for h in self._holdings.values())
        else:
            total_value = total_cost

        total_return = total_value - total_cost
        return_pct = (total_return / total_cost * 100) if total_cost > 0 else 0

        # Find top holding by value
        values = {}
        for h in self._holdings.values():
            price = current_prices.get(h.ticker, h.avg_cost) if current_prices else h.avg_cost
            values[h.ticker] = h.shares * price

        top = max(values, key=values.get) if values else ""
        top_weight = (values[top] / total_value * 100) if total_value > 0 and top else 0

        return PortfolioSummary(
            total_cost=total_cost,
            total_value=total_value,
            total_return=total_return,
            return_pct=return_pct,
            holdings_count=len(self._holdings),
            top_holding=top,
            top_weight=top_weight,
        )

    def allocation(self, current_prices: dict[str, float] | None = None) -> list[dict]:
        """Get portfolio allocation breakdown."""
        if not self._holdings:
            return []

        values = {}
        for h in self._holdings.values():
            price = current_prices.get(h.ticker, h.avg_cost) if current_prices else h.avg_cost
            values[h.ticker] = h.shares * price

        total = sum(values.values())
        result = []
        for ticker, value in sorted(values.items(), key=lambda x: x[1], reverse=True):
            h = self._holdings[ticker]
            result.append(
                {
                    "ticker": ticker,
                    "shares": h.shares,
                    "value": round(value, 2),
                    "weight_pct": round(value / total * 100, 1) if total > 0 else 0,
                    "avg_cost": round(h.avg_cost, 2),
                }
            )
        return result

    def save(self, path: str | Path) -> None:
        data = {
            "name": self.name,
            "created_at": self._created_at,
            "holdings": {t: h.to_dict() for t, h in self._holdings.items()},
        }
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")

    def load(self, path: str | Path) -> None:
        p = Path(path)
        if not p.exists():
            return
        data = json.loads(p.read_text(encoding="utf-8"))
        self.name = data.get("name", self.name)
        self._created_at = data.get("created_at", self._created_at)
        for _ticker, hdata in data.get("holdings", {}).items():
            self._holdings[hdata["ticker"]] = Holding(
                ticker=hdata["ticker"],
                shares=hdata["shares"],
                avg_cost=hdata["avg_cost"],
                added_at=hdata.get("added_at", 0),
            )
