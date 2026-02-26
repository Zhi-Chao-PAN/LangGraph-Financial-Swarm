"""Tests for trend analysis module."""

from src.utils.trend import (
    _detect_anomalies,
    _linear_regression,
    _moving_average,
    analyze_multiple_trends,
    analyze_trend,
)


class TestLinearRegression:
    def test_perfect_positive(self):
        slope, intercept, r2 = _linear_regression([0, 1, 2, 3], [0, 1, 2, 3])
        assert abs(slope - 1.0) < 0.01
        assert abs(r2 - 1.0) < 0.01

    def test_perfect_negative(self):
        slope, intercept, r2 = _linear_regression([0, 1, 2, 3], [3, 2, 1, 0])
        assert slope < 0
        assert abs(r2 - 1.0) < 0.01

    def test_flat(self):
        slope, intercept, r2 = _linear_regression([0, 1, 2, 3], [5, 5, 5, 5])
        assert abs(slope) < 0.01

    def test_single_point(self):
        slope, intercept, r2 = _linear_regression([0], [5])
        assert slope == 0.0

    def test_two_points(self):
        slope, intercept, r2 = _linear_regression([0, 1], [10, 20])
        assert abs(slope - 10.0) < 0.01


class TestDetectAnomalies:
    def test_no_anomalies(self):
        assert _detect_anomalies([10, 11, 12, 13, 14]) == []

    def test_with_anomaly(self):
        values = [10, 11, 12, 100, 13]
        anomalies = _detect_anomalies(values)
        assert 3 in anomalies

    def test_too_few_values(self):
        assert _detect_anomalies([1, 2]) == []

    def test_all_same(self):
        assert _detect_anomalies([5, 5, 5, 5]) == []


class TestMovingAverage:
    def test_basic(self):
        result = _moving_average([1, 2, 3, 4, 5], window=3)
        assert len(result) == 5
        assert result[-1] == 4.0  # (3+4+5)/3

    def test_short_list(self):
        result = _moving_average([1, 2], window=3)
        assert result == [1, 2]


class TestAnalyzeTrend:
    def test_upward(self):
        result = analyze_trend("Revenue", ["2021", "2022", "2023", "2024"], [100, 150, 200, 250])
        assert result.direction == "up"
        assert result.slope > 0
        assert result.r_squared > 0.9
        assert result.forecast_next is not None
        assert result.forecast_next > 250

    def test_downward(self):
        result = analyze_trend("Margin", ["2021", "2022", "2023", "2024"], [80, 60, 40, 20])
        assert result.direction == "down"
        assert result.slope < 0

    def test_flat(self):
        result = analyze_trend("Ratio", ["2021", "2022", "2023"], [50, 50, 50])
        assert result.direction == "flat"

    def test_single_value(self):
        result = analyze_trend("Revenue", ["2024"], [100])
        assert result.direction == "flat"
        assert result.slope == 0.0

    def test_no_forecast(self):
        result = analyze_trend("Revenue", ["2021", "2022"], [100, 200], forecast=False)
        assert result.forecast_next is None

    def test_summary(self):
        result = analyze_trend("Revenue", ["2021", "2022", "2023"], [100, 200, 300])
        s = result.summary()
        assert "Revenue" in s
        assert "up" in s.lower() or "↑" in s

    def test_is_significant(self):
        result = analyze_trend("Revenue", ["2021", "2022", "2023", "2024"], [100, 200, 300, 400])
        assert result.is_significant

    def test_anomaly_detection(self):
        result = analyze_trend("Revenue", ["2021", "2022", "2023", "2024", "2025"], [100, 110, 120, 500, 130])
        assert result.anomalies is not None
        assert 3 in result.anomalies


class TestAnalyzeMultipleTrends:
    def test_basic(self):
        data = {
            "2021": {"Revenue": 100, "Profit": 20},
            "2022": {"Revenue": 150, "Profit": 30},
            "2023": {"Revenue": 200, "Profit": 40},
        }
        results = analyze_multiple_trends(data, ["Revenue", "Profit"])
        assert len(results) == 2
        assert all(r.direction == "up" for r in results)

    def test_missing_metric(self):
        data = {
            "2021": {"Revenue": 100},
            "2022": {"Revenue": 150},
        }
        results = analyze_multiple_trends(data, ["Revenue", "NonExistent"])
        assert len(results) == 1

    def test_partial_data(self):
        data = {
            "2021": {"Revenue": 100},
            "2022": {},
            "2023": {"Revenue": 200},
        }
        results = analyze_multiple_trends(data, ["Revenue"])
        assert len(results) == 1
        assert len(results[0].values) == 2
