"""Tests for query template system."""

from src.core.templates import (
    TEMPLATE_CATEGORIES,
    TEMPLATE_MAP,
    TEMPLATES,
    generate_batch_queries,
    get_template,
    list_templates,
    render_template,
)


class TestQueryTemplate:
    def test_render_basic(self):
        t = get_template("revenue")
        assert t is not None
        result = t.render(company="NVIDIA", year="FY2025")
        assert "NVIDIA" in result
        assert "FY2025" in result

    def test_render_comparison(self):
        t = get_template("compare_revenue")
        result = t.render(company1="NVIDIA", company2="Apple", year="FY2025")
        assert "NVIDIA" in result
        assert "Apple" in result

    def test_missing_params(self):
        t = get_template("revenue")
        assert set(t.missing_params) == {"company", "year"}
        rendered = t.render(company="NVIDIA")
        # year still in template
        assert "{year}" in rendered

    def test_all_templates_have_params(self):
        for t in TEMPLATES:
            assert len(t.params) > 0
            assert t.name
            assert t.category


class TestTemplateRegistry:
    def test_template_map(self):
        assert "revenue" in TEMPLATE_MAP
        assert "cagr" in TEMPLATE_MAP

    def test_categories(self):
        assert "single_hop" in TEMPLATE_CATEGORIES
        assert "comparison" in TEMPLATE_CATEGORIES
        assert "calculation" in TEMPLATE_CATEGORIES
        assert "trend" in TEMPLATE_CATEGORIES

    def test_list_all(self):
        all_templates = list_templates()
        assert len(all_templates) >= 10

    def test_list_by_category(self):
        single = list_templates("single_hop")
        assert all(t.category == "single_hop" for t in single)
        assert len(single) >= 3

    def test_list_empty_category(self):
        result = list_templates("nonexistent")
        assert result == []

    def test_get_nonexistent(self):
        assert get_template("nonexistent") is None

    def test_render_template_helper(self):
        result = render_template("revenue", company="Apple", year="FY2024")
        assert "Apple" in result
        assert "FY2024" in result

    def test_render_template_nonexistent(self):
        assert render_template("nonexistent") is None


class TestBatchGeneration:
    def test_basic_batch(self):
        queries = generate_batch_queries(
            companies=["NVIDIA", "Apple"],
            years=["FY2024", "FY2025"],
            template_names=["revenue"],
        )
        assert len(queries) == 4  # 2 companies × 2 years
        assert all("revenue" in q.lower() for q in queries)

    def test_comparison_batch(self):
        queries = generate_batch_queries(
            companies=["NVIDIA", "Apple", "Microsoft"],
            years=["FY2025"],
            template_names=["compare_revenue"],
        )
        # C(3,2) × 1 year = 3
        assert len(queries) == 3

    def test_trend_batch(self):
        queries = generate_batch_queries(
            companies=["NVIDIA"],
            years=["FY2023", "FY2024", "FY2025"],
            template_names=["revenue_growth"],
        )
        # 1 company × C(3,2) year pairs = 3
        assert len(queries) == 3

    def test_empty_companies(self):
        queries = generate_batch_queries(companies=[], years=["FY2025"], template_names=["revenue"])
        assert queries == []

    def test_multi_template_batch(self):
        queries = generate_batch_queries(
            companies=["NVIDIA"],
            years=["FY2025"],
            template_names=["revenue", "net_income", "gross_margin"],
        )
        assert len(queries) == 3
