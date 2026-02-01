# src/tools/rag_tool.py
import os
from typing import Optional
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool

from src.rag_adapter import adapter

def query_rag_engine(question: str) -> str:
    """
    Interface with the Structure-Aware RAG Engine via Adapter.
    """
    result = adapter.query(question)
    return result["model_answer"]

@tool
def query_financial_rag(question: str) -> str:
    """
    Search and retrieve precise data from historical financial reports.
    This tool is structure-aware and can handle complex tables and financial statements.
    Use this for any factual inquiries about company performance or metrics.
    """
    return query_rag_engine(question)
