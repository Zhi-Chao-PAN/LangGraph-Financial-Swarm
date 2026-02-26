"""
FastAPI REST API server for LangGraph Financial Swarm.
Provides endpoints for querying, fetching data, generating reports, and system status.

Usage:
    uvicorn src.api.server:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(
    title="LangGraph Financial Swarm API",
    description="Multi-agent financial analysis system with RAG",
    version="0.9.0",
)


# 鈹€鈹€鈹€ Request/Response Models 鈹€鈹€鈹€


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Financial question to answer")
    language: str = Field("en", description="Response language (en/zh)")


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[str]
    latency: float
    confidence: float | None = None


class FetchRequest(BaseModel):
    tickers: list[str] = Field(..., min_length=1, description="Stock ticker(s) to fetch")
    force: bool = Field(False, description="Force refresh (ignore cache)")


class FetchResponse(BaseModel):
    results: list[dict[str, Any]]


class ReportRequest(BaseModel):
    ticker: str = Field(..., description="Stock ticker")
    language: str = Field("en", description="Report language (en/zh)")


class ReportResponse(BaseModel):
    ticker: str
    report: str
    format: str = "markdown"


class CompareRequest(BaseModel):
    tickers: list[str] = Field(..., min_length=2, description="Tickers to compare")
    ratios: list[str] | None = Field(None, description="Specific ratios to compare")


class HealthResponse(BaseModel):
    status: str
    version: str
    llm_available: bool
    data_sources: int


# 鈹€鈹€鈹€ Endpoints 鈹€鈹€鈹€


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """System health check."""
    from src.core.llm_backends import PRESETS, check_backend_available
    from src.data.sources import registry

    llm_ok = any(check_backend_available(cfg) for cfg in PRESETS.values())

    return HealthResponse(
        status="ok",
        version="0.9.0",
        llm_available=llm_ok,
        data_sources=len(registry.list_available()),
    )


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    """Ask a financial question."""
    import time

    try:
        from src.rag_adapter import adapter

        if adapter.index is None:
            adapter._build_index()

        start = time.time()
        result = adapter.query(req.question)
        latency = time.time() - start

        answer = result.get("model_answer", "No answer found.")
        sources = [str(s) for s in result.get("source_nodes", [])]

        # Extract confidence if present
        confidence = None
        import re

        match = re.search(r"\[Confidence:\s*([\d.]+)\]", answer)
        if match:
            confidence = float(match.group(1))

        return QueryResponse(
            question=req.question,
            answer=answer,
            sources=sources[:5],
            latency=latency,
            confidence=confidence,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/fetch", response_model=FetchResponse)
async def fetch_data(req: FetchRequest):
    """Fetch financial data for tickers."""
    from src.data.finance_fetcher import fetch_company_data

    results = []
    for ticker in req.tickers:
        try:
            data = fetch_company_data(ticker)
            results.append(
                {
                    "ticker": ticker,
                    "name": data["name"],
                    "status": "ok",
                    "info": data["info"],
                }
            )
        except Exception as e:
            results.append(
                {
                    "ticker": ticker,
                    "status": "error",
                    "error": str(e),
                }
            )

    return FetchResponse(results=results)


@app.post("/report", response_model=ReportResponse)
async def generate_report_endpoint(req: ReportRequest):
    """Generate a financial health report."""
    from src.data.finance_fetcher import fetch_company_data
    from src.utils.report import ReportConfig, generate_report

    try:
        data = fetch_company_data(req.ticker)
        latest_year = max(data["income_stmt"].keys()) if data["income_stmt"] else None

        income = data["income_stmt"].get(latest_year, {}) if latest_year else {}
        balance = data["balance_sheet"].get(latest_year, {}) if latest_year else {}
        cashflow = data["cashflow"].get(latest_year, {}) if latest_year else {}

        config = ReportConfig(language=req.language)
        report = generate_report(
            company_name=data["name"],
            ticker=req.ticker.upper(),
            income=income,
            balance=balance,
            cashflow=cashflow,
            historical=data["income_stmt"],
            config=config,
        )

        return ReportResponse(
            ticker=req.ticker.upper(),
            report=report,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/compare")
async def compare_companies(req: CompareRequest):
    """Compare multiple companies."""
    from src.data.finance_fetcher import fetch_company_data
    from src.utils.peer_compare import compare_peers

    peer_data = {}
    for ticker in req.tickers:
        try:
            data = fetch_company_data(ticker)
            latest_year = max(data["income_stmt"].keys()) if data["income_stmt"] else None
            if latest_year:
                peer_data[ticker.upper()] = {
                    "income": data["income_stmt"].get(latest_year, {}),
                    "balance": data["balance_sheet"].get(latest_year, {}),
                    "cashflow": data["cashflow"].get(latest_year, {}),
                }
        except Exception:
            continue

    if len(peer_data) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 companies with data")

    result = compare_peers(req.tickers[0], peer_data, ratios=req.ratios)
    return {"markdown": result.to_markdown(), "target": result.target_ticker, "peers": len(result.peers)}


@app.get("/backends")
async def list_llm_backends():
    """List available LLM backends."""
    from src.core.llm_backends import list_backends

    return {"backends": list_backends()}
