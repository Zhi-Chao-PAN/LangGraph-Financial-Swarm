"""
Trend analysis and simple forecasting module.
Uses linear regression and moving averages for financial trend detection.
"""

from dataclasses import dataclass


@dataclass
class TrendResult:
    """Result of a trend analysis."""

    metric: str
    direction: str  # "up", "down", "flat"
    slope: float
    r_squared: float
    values: list[float]
    years: list[str]
    forecast_next: float | None = None
    anomalies: list[int] | None = None  # indices of anomalous years

    @property
    def is_significant(self) -> bool:
        return self.r_squared >= 0.5

    def summary(self) -> str:
        arrow = {"up": "↑", "down": "↓", "flat": "→"}.get(self.direction, "?")
        sig = "strong" if self.r_squared >= 0.8 else "moderate" if self.r_squared >= 0.5 else "weak"
        msg = f"{self.metric}: {arrow} {self.direction} ({sig}, R²={self.r_squared:.2f})"
        if self.forecast_next is not None:
            msg += f", forecast next: {self.forecast_next:,.0f}"
        if self.anomalies:
            anomaly_years = [self.years[i] for i in self.anomalies if i < len(self.years)]
            msg += f", anomalies: {anomaly_years}"
        return msg


def _linear_regression(x: list[float], y: list[float]) -> tuple[float, float, float]:
    """
    Simple linear regression. Returns (slope, intercept, r_squared).
    No external dependencies needed.
    """
    n = len(x)
    if n < 2:
        return 0.0, 0.0, 0.0

    sum_x = sum(x)
    sum_y = sum(y)
    sum_xy = sum(xi * yi for xi, yi in zip(x, y, strict=False))
    sum_x2 = sum(xi**2 for xi in x)

    denom = n * sum_x2 - sum_x**2
    if denom == 0:
        return 0.0, sum_y / n, 0.0

    slope = (n * sum_xy - sum_x * sum_y) / denom
    intercept = (sum_y - slope * sum_x) / n

    # R-squared
    ss_res = sum((yi - (slope * xi + intercept)) ** 2 for xi, yi in zip(x, y, strict=False))
    mean_y = sum_y / n
    ss_tot = sum((yi - mean_y) ** 2 for yi in y)

    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
    return slope, intercept, max(0.0, r_squared)


def _detect_anomalies(values: list[float], threshold: float = 1.5) -> list[int]:
    """Detect anomalies using Z-score method."""
    if len(values) < 3:
        return []

    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    std = variance**0.5

    if std == 0:
        return []

    return [i for i, v in enumerate(values) if abs((v - mean) / std) > threshold]


def _moving_average(values: list[float], window: int = 3) -> list[float]:
    """Calculate moving average."""
    if len(values) < window:
        return values[:]
    result = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        result.append(sum(values[start : i + 1]) / (i - start + 1))
    return result


def analyze_trend(
    metric: str,
    years: list[str],
    values: list[float],
    forecast: bool = True,
) -> TrendResult:
    """
    Analyze the trend of a financial metric over time.

    Args:
        metric: Name of the metric
        years: List of year labels
        values: Corresponding numeric values
        forecast: Whether to forecast the next period
    """
    if len(values) < 2:
        return TrendResult(
            metric=metric,
            direction="flat",
            slope=0.0,
            r_squared=0.0,
            values=values,
            years=years,
        )

    x = list(range(len(values)))
    slope, intercept, r_squared = _linear_regression(
        [float(xi) for xi in x],
        [float(v) for v in values],
    )

    # Determine direction
    if abs(slope) < 0.01 * abs(sum(values) / len(values)):
        direction = "flat"
    elif slope > 0:
        direction = "up"
    else:
        direction = "down"

    # Forecast
    forecast_next = None
    if forecast:
        forecast_next = slope * len(values) + intercept

    # Anomaly detection
    anomalies = _detect_anomalies(values)

    return TrendResult(
        metric=metric,
        direction=direction,
        slope=slope,
        r_squared=r_squared,
        values=values,
        years=years,
        forecast_next=forecast_next,
        anomalies=anomalies if anomalies else None,
    )


def analyze_multiple_trends(
    data: dict[str, dict[str, float]],
    metrics: list[str],
) -> list[TrendResult]:
    """
    Analyze trends for multiple metrics across years.

    Args:
        data: Dict of year -> {metric: value}
        metrics: List of metric names to analyze
    """
    years = sorted(data.keys())
    results = []

    for metric in metrics:
        values = []
        valid_years = []
        for y in years:
            v = data.get(y, {}).get(metric)
            if v is not None:
                values.append(float(v))
                valid_years.append(y)

        if len(values) >= 2:
            results.append(analyze_trend(metric, valid_years, values))

    return results
