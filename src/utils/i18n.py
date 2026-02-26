"""
Multi-language support for financial queries.
Detects language, maps Chinese financial terms to English,
and formats responses in the user's language.
"""

import re

# Chinese to English financial term mapping
ZH_EN_TERMS: dict[str, str] = {
    # Income Statement
    "营收": "revenue",
    "收入": "revenue",
    "总收入": "total revenue",
    "营业收入": "total revenue",
    "净利润": "net income",
    "净收入": "net income",
    "毛利": "gross profit",
    "毛利润": "gross profit",
    "营业利润": "operating income",
    "运营利润": "operating income",
    "研发费用": "research and development",
    "研发支出": "research and development",
    "销售费用": "selling general and administration",
    "每股收益": "diluted EPS",
    "每股盈利": "diluted EPS",
    # Ratios
    "毛利率": "gross margin",
    "营业利润率": "operating margin",
    "净利率": "net margin",
    "净利润率": "net margin",
    "资产回报率": "ROA",
    "净资产收益率": "ROE",
    "研发占比": "R&D ratio",
    "负债率": "debt to assets",
    "资产负债率": "debt to assets",
    "流动比率": "current ratio",
    "速动比率": "quick ratio",
    "市值": "market cap",
    "市盈率": "P/E ratio",
    # Balance Sheet
    "总资产": "total assets",
    "总负债": "total liabilities",
    "股东权益": "stockholders equity",
    "现金": "cash and cash equivalents",
    "总债务": "total debt",
    "净债务": "net debt",
    # Cash Flow
    "经营现金流": "operating cash flow",
    "自由现金流": "free cash flow",
    "资本支出": "capital expenditure",
    # Segments
    "云业务": "cloud",
    "广告收入": "advertising revenue",
    # Actions
    "对比": "compare",
    "比较": "compare",
    "趋势": "trend",
    "增长": "growth",
    "下降": "decline",
    "分析": "analyze",
    "报告": "report",
    # Companies
    "英伟达": "NVIDIA",
    "苹果": "Apple",
    "微软": "Microsoft",
    "谷歌": "Alphabet",
    "亚马逊": "Amazon",
    "特斯拉": "Tesla",
    "阿里巴巴": "Alibaba",
    "腾讯": "Tencent",
    "百度": "Baidu",
    "京东": "JD",
    "拼多多": "PDD",
    # Time
    "财年": "FY",
    "去年": "last year",
    "今年": "this year",
    "季度": "quarter",
    "年度": "annual",
}

# English to Chinese term mapping (for response translation)
EN_ZH_TERMS: dict[str, str] = {
    "revenue": "营收",
    "total revenue": "总收入",
    "net income": "净利润",
    "gross profit": "毛利润",
    "operating income": "营业利润",
    "gross margin": "毛利率",
    "operating margin": "营业利润率",
    "net margin": "净利率",
    "total assets": "总资产",
    "total debt": "总债务",
    "free cash flow": "自由现金流",
    "operating cash flow": "经营现金流",
    "research and development": "研发费用",
}


def detect_language(text: str) -> str:
    """Detect if text is primarily Chinese or English."""
    chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
    total_chars = len(text.strip())
    if total_chars == 0:
        return "en"
    if chinese_chars / total_chars > 0.15:
        return "zh"
    return "en"


def translate_query(query: str) -> str:
    """
    Translate a Chinese financial query to English for RAG retrieval.
    Preserves numbers, tickers, and English terms.
    """
    if detect_language(query) != "zh":
        return query

    translated = query
    # Sort by length (longest first) to avoid partial matches
    sorted_terms = sorted(ZH_EN_TERMS.items(), key=lambda x: len(x[0]), reverse=True)
    for zh, en in sorted_terms:
        translated = translated.replace(zh, en)

    # Clean up: remove remaining Chinese chars that weren't mapped
    # but keep the translated parts
    return translated.strip()


def localize_response(response: str, target_lang: str) -> str:
    """
    Localize key financial terms in a response to the target language.
    Only translates known financial terms, preserves numbers and structure.
    """
    if target_lang == "en":
        return response

    localized = response
    # Sort by length (longest first)
    sorted_terms = sorted(EN_ZH_TERMS.items(), key=lambda x: len(x[0]), reverse=True)
    for en, zh in sorted_terms:
        # Case-insensitive replacement for known terms
        pattern = re.compile(re.escape(en), re.IGNORECASE)
        localized = pattern.sub(zh, localized)

    return localized
