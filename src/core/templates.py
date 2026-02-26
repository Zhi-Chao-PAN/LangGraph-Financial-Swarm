"""
Query template system for common financial analysis patterns.
Provides pre-built templates that users can fill with company/year parameters.
"""

from dataclasses import dataclass, field


@dataclass
class QueryTemplate:
    """A parameterized query template."""

    name: str
    description: str
    template: str
    category: str
    params: list[str] = field(default_factory=list)
    difficulty: str = "easy"

    def render(self, **kwargs: str) -> str:
        """Render the template with given parameters."""
        result = self.template
        for key, value in kwargs.items():
            result = result.replace(f"{{{key}}}", value)
        return result

    @property
    def missing_params(self) -> list[str]:
        """List parameters still in the template."""
        import re

        return re.findall(r"\{(\w+)\}", self.template)


# ─── Built-in Templates ───

TEMPLATES: list[QueryTemplate] = [
    # Single-hop
    QueryTemplate(
        name="revenue",
        description="Get company revenue for a fiscal year",
        template="What was {company}'s revenue in {year}?",
        category="single_hop",
        params=["company", "year"],
    ),
    QueryTemplate(
        name="net_income",
        description="Get company net income for a fiscal year",
        template="What was {company}'s net income in {year}?",
        category="single_hop",
        params=["company", "year"],
    ),
    QueryTemplate(
        name="gross_margin",
        description="Get company gross margin for a fiscal year",
        template="What was {company}'s gross margin in {year}?",
        category="single_hop",
        params=["company", "year"],
    ),
    QueryTemplate(
        name="operating_income",
        description="Get company operating income for a fiscal year",
        template="What was {company}'s operating income in {year}?",
        category="single_hop",
        params=["company", "year"],
    ),
    # Comparison
    QueryTemplate(
        name="compare_revenue",
        description="Compare revenue between two companies",
        template="Compare {company1} and {company2} revenue in {year}",
        category="comparison",
        params=["company1", "company2", "year"],
        difficulty="medium",
    ),
    QueryTemplate(
        name="compare_margin",
        description="Compare gross margins between companies",
        template="Which company had higher gross margin in {year}, {company1} or {company2}?",
        category="comparison",
        params=["company1", "company2", "year"],
        difficulty="medium",
    ),
    QueryTemplate(
        name="highest_revenue",
        description="Find which company had the highest revenue",
        template="Which company had the highest revenue in {year}?",
        category="comparison",
        params=["year"],
        difficulty="medium",
    ),
    # Calculation
    QueryTemplate(
        name="revenue_growth",
        description="Calculate revenue growth between two years",
        template="What was {company}'s revenue growth from {year1} to {year2}?",
        category="calculation",
        params=["company", "year1", "year2"],
        difficulty="medium",
    ),
    QueryTemplate(
        name="cagr",
        description="Calculate compound annual growth rate",
        template="Calculate {company}'s revenue CAGR from {year1} to {year2}",
        category="calculation",
        params=["company", "year1", "year2"],
        difficulty="hard",
    ),
    QueryTemplate(
        name="debt_equity",
        description="Calculate debt-to-equity ratio",
        template="What was {company}'s debt-to-equity ratio in {year}?",
        category="calculation",
        params=["company", "year"],
        difficulty="hard",
    ),
    # Trend
    QueryTemplate(
        name="revenue_trend",
        description="Analyze revenue trend over a period",
        template="What is {company}'s revenue trend from {year1} to {year2}?",
        category="trend",
        params=["company", "year1", "year2"],
        difficulty="hard",
    ),
    QueryTemplate(
        name="margin_trend",
        description="Analyze margin trend over a period",
        template="How has {company}'s gross margin changed from {year1} to {year2}?",
        category="trend",
        params=["company", "year1", "year2"],
        difficulty="hard",
    ),
]

# Index by name
TEMPLATE_MAP: dict[str, QueryTemplate] = {t.name: t for t in TEMPLATES}

# Index by category
TEMPLATE_CATEGORIES: dict[str, list[QueryTemplate]] = {}
for _t in TEMPLATES:
    TEMPLATE_CATEGORIES.setdefault(_t.category, []).append(_t)


def get_template(name: str) -> QueryTemplate | None:
    """Get a template by name."""
    return TEMPLATE_MAP.get(name)


def list_templates(category: str | None = None) -> list[QueryTemplate]:
    """List available templates, optionally filtered by category."""
    if category:
        return TEMPLATE_CATEGORIES.get(category, [])
    return TEMPLATES


def render_template(name: str, **kwargs: str) -> str | None:
    """Render a template by name with given parameters."""
    template = get_template(name)
    if not template:
        return None
    return template.render(**kwargs)


def generate_batch_queries(
    companies: list[str],
    years: list[str],
    template_names: list[str] | None = None,
) -> list[str]:
    """
    Generate a batch of queries from templates for given companies and years.
    Useful for systematic evaluation or data collection.
    """
    templates = [TEMPLATE_MAP[n] for n in template_names if n in TEMPLATE_MAP] if template_names else TEMPLATES
    queries = []

    for t in templates:
        params = t.params
        if "company" in params and "year" in params:
            for company in companies:
                for year in years:
                    queries.append(t.render(company=company, year=year))
        elif "company1" in params and "company2" in params:
            for i, c1 in enumerate(companies):
                for c2 in companies[i + 1 :]:
                    for year in years:
                        queries.append(t.render(company1=c1, company2=c2, year=year))
        elif "company" in params and "year1" in params and "year2" in params:
            for company in companies:
                for i, y1 in enumerate(years):
                    for y2 in years[i + 1 :]:
                        queries.append(t.render(company=company, year1=y1, year2=y2))

    return queries
