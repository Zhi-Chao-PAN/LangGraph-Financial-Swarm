"""Tests for FastAPI REST API server."""

from unittest.mock import patch

from fastapi.testclient import TestClient

from src.api.server import app

client = TestClient(app)


class TestHealthEndpoint:
    @patch("src.core.llm_backends.check_backend_available", return_value=False)
    def test_health(self, mock_check):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "version" in data
        assert "llm_available" in data


class TestQueryEndpoint:
    def test_query_empty(self):
        resp = client.post("/query", json={"question": ""})
        assert resp.status_code == 422  # Validation error

    @patch("src.rag_adapter.adapter")
    def test_query_success(self, mock_adapter):
        mock_adapter.index = True
        mock_adapter.query.return_value = {
            "model_answer": "Revenue was $130B [Confidence: 0.95]",
            "source_nodes": ["src1"],
        }
        resp = client.post("/query", json={"question": "What was NVIDIA revenue?"})
        assert resp.status_code == 200
        data = resp.json()
        assert "answer" in data
        assert data["confidence"] == 0.95


class TestFetchEndpoint:
    @patch("src.data.finance_fetcher.fetch_company_data")
    def test_fetch(self, mock_fetch):
        mock_fetch.return_value = {
            "name": "Apple Inc.",
            "ticker": "AAPL",
            "info": {
                "sector": "Tech",
                "industry": "CE",
                "country": "US",
                "currency": "USD",
            },
            "income_stmt": {},
            "balance_sheet": {},
            "cashflow": {},
        }
        resp = client.post("/fetch", json={"tickers": ["AAPL"]})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["results"]) == 1
        assert data["results"][0]["status"] == "ok"

    def test_fetch_empty(self):
        resp = client.post("/fetch", json={"tickers": []})
        assert resp.status_code == 422


class TestReportEndpoint:
    @patch("src.data.finance_fetcher.fetch_company_data")
    def test_report(self, mock_fetch):
        mock_fetch.return_value = {
            "name": "NVIDIA",
            "ticker": "NVDA",
            "info": {
                "sector": "Tech",
                "industry": "Semi",
                "country": "US",
                "currency": "USD",
            },
            "income_stmt": {
                "2024": {
                    "Total Revenue": 130_000,
                    "Net Income": 72_000,
                    "Gross Profit": 100_000,
                },
            },
            "balance_sheet": {"2024": {"Total Assets": 200_000}},
            "cashflow": {"2024": {"Free Cash Flow": 60_000}},
        }
        resp = client.post("/report", json={"ticker": "NVDA"})
        assert resp.status_code == 200
        data = resp.json()
        assert "NVDA" in data["ticker"]
        assert "report" in data

    @patch("src.data.finance_fetcher.fetch_company_data")
    def test_report_chinese(self, mock_fetch):
        mock_fetch.return_value = {
            "name": "NVIDIA",
            "ticker": "NVDA",
            "info": {
                "sector": "Tech",
                "industry": "Semi",
                "country": "US",
                "currency": "USD",
            },
            "income_stmt": {"2024": {"Total Revenue": 130_000}},
            "balance_sheet": {},
            "cashflow": {},
        }
        resp = client.post(
            "/report", json={"ticker": "NVDA", "language": "zh"}
        )
        assert resp.status_code == 200


class TestBackendsEndpoint:
    @patch("src.core.llm_backends.list_backends")
    def test_backends(self, mock_list):
        mock_list.return_value = [{"name": "test", "available": True}]
        resp = client.get("/backends")
        assert resp.status_code == 200
        assert "backends" in resp.json()
