"""
Streamlit Web UI for LangGraph Financial Swarm.
Provides a browser-based interface for financial analysis queries.

Usage:
    streamlit run src/ui/web.py
"""

import re
import time
from pathlib import Path

import streamlit as st

# Page config must be first Streamlit command
st.set_page_config(
    page_title="Financial Swarm",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _get_settings():
    """Lazy import to avoid circular imports."""
    from src.core.config import settings

    return settings


def _get_health():
    from src.utils.robustness import check_ollama_health

    return check_ollama_health()


def _get_adapter():
    from src.rag_adapter import adapter

    return adapter


# ─── Sidebar ───


def render_sidebar():
    """Render the sidebar with system status and settings."""
    st.sidebar.title("🏦 Financial Swarm")
    st.sidebar.caption("v0.9.0 · Multi-Agent Financial Analysis")
    st.sidebar.divider()

    # System status
    st.sidebar.subheader("System Status")

    settings = _get_settings()
    health = _get_health()

    if health["healthy"]:
        st.sidebar.success(f"Ollama: Online ({len(health['models'])} models)")
    else:
        st.sidebar.error(f"Ollama: Offline — {health.get('error', 'Unknown')}")

    st.sidebar.info(f"Model: {settings.LLM_MODEL}")
    st.sidebar.info(f"Embedding: {settings.EMBEDDING_MODEL}")

    rag_path = Path(settings.RAG_DATA_PATH)
    if rag_path.exists():
        size_kb = rag_path.stat().st_size / 1024
        st.sidebar.success(f"RAG Data: {rag_path.name} ({size_kb:.0f} KB)")
    else:
        st.sidebar.warning("RAG Data: Not found")

    st.sidebar.divider()

    # Session stats
    st.sidebar.subheader("Session Stats")
    stats = st.session_state.get("stats", {"queries": 0, "avg_latency": 0})
    col1, col2 = st.sidebar.columns(2)
    col1.metric("Queries", stats["queries"])
    col2.metric("Avg Latency", f"{stats['avg_latency']:.1f}s")

    st.sidebar.divider()

    # Cache management
    if st.sidebar.button("🗑️ Clear Cache"):
        import shutil

        cache_dir = settings.DATA_DIR / "rag_cache"
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
        st.sidebar.success("Cache cleared!")

    # Export
    if st.session_state.get("history"):
        import json

        export_data = json.dumps(st.session_state["history"], indent=2, ensure_ascii=False)
        st.sidebar.download_button(
            "📥 Export History (JSON)",
            data=export_data,
            file_name="swarm_history.json",
            mime="application/json",
        )


# ─── Main Content ───


def render_query_section():
    """Render the main query input and results area."""
    st.header("💬 Ask a Financial Question")

    # Example questions
    examples = [
        "What was NVIDIA's revenue in FY2025?",
        "Compare Apple and Microsoft net income in FY2024",
        "Which company had the highest gross margin?",
        "Calculate NVIDIA's revenue CAGR from FY2021 to FY2025",
    ]

    col1, col2 = st.columns([3, 1])
    with col1:
        question = st.text_input(
            "Question",
            placeholder="e.g., What was NVIDIA's revenue in FY2025?",
            label_visibility="collapsed",
        )
    with col2:
        submit = st.button("🔍 Ask", type="primary", use_container_width=True)

    # Quick examples
    st.caption("Try an example:")
    example_cols = st.columns(len(examples))
    for i, ex in enumerate(examples):
        if example_cols[i].button(ex[:35] + "...", key=f"ex_{i}", use_container_width=True):
            question = ex
            submit = True

    if submit and question:
        _execute_and_display(question)


def _execute_and_display(question: str):
    """Execute query and display results."""
    adapter = _get_adapter()

    with st.spinner("🔍 Searching financial data..."):
        start = time.time()
        try:
            result = adapter.query(question)
            latency = time.time() - start
        except Exception as e:
            st.error(f"Query failed: {e}")
            return

    answer = result.get("model_answer", "No answer found.")

    # Extract confidence
    confidence = None
    if "[Confidence:" in answer:
        match = re.search(r"\[Confidence:\s*([\d.]+)\]", answer)
        if match:
            confidence = float(match.group(1))
            answer = re.sub(r"\s*\[Confidence:.*?\]", "", answer)

    # Split answer and citations
    parts = answer.split("**Citations**:")
    main_answer = parts[0].strip()
    citations = parts[1].strip() if len(parts) > 1 else ""

    # Display answer
    st.subheader("📊 Answer")

    # Metrics row
    metric_cols = st.columns(3)
    metric_cols[0].metric("Latency", f"{latency:.1f}s")
    if confidence is not None:
        metric_cols[1].metric("Confidence", f"{confidence:.0%}")
    sources_count = len(result.get("source_nodes", []))
    metric_cols[2].metric("Sources", sources_count)

    # Main answer
    st.markdown(main_answer)

    # Citations
    if citations:
        with st.expander("📚 Citations", expanded=False):
            for line in citations.strip().split("\n"):
                if line.strip():
                    st.caption(line.strip())

    # Source documents
    source_nodes = result.get("source_nodes", [])
    if source_nodes:
        with st.expander(f"📄 Source Documents ({len(source_nodes)})", expanded=False):
            for i, src in enumerate(source_nodes):
                st.text_area(f"Source {i + 1}", src[:500], height=100, disabled=True, key=f"src_{time.time()}_{i}")

    # Update session state
    if "history" not in st.session_state:
        st.session_state["history"] = []
    if "stats" not in st.session_state:
        st.session_state["stats"] = {"queries": 0, "total_latency": 0, "avg_latency": 0}

    st.session_state["history"].append(
        {
            "question": question,
            "answer": main_answer,
            "confidence": confidence,
            "latency": latency,
            "sources": sources_count,
        }
    )

    stats = st.session_state["stats"]
    stats["queries"] += 1
    stats["total_latency"] += latency
    stats["avg_latency"] = stats["total_latency"] / stats["queries"]


def render_history():
    """Render conversation history."""
    history = st.session_state.get("history", [])
    if not history:
        return

    st.divider()
    st.subheader("📜 Conversation History")

    for i, h in enumerate(reversed(history[:-1]), 1):  # Skip most recent (already shown)
        with st.expander(f"Q{len(history) - i}: {h['question'][:80]}", expanded=False):
            st.markdown(h["answer"][:500])
            cols = st.columns(3)
            cols[0].caption(f"Latency: {h['latency']:.1f}s")
            if h.get("confidence"):
                cols[1].caption(f"Confidence: {h['confidence']:.0%}")
            cols[2].caption(f"Sources: {h.get('sources', 'N/A')}")


# ─── Main ───


def main():
    """Main Streamlit app entry point."""
    # Initialize session state
    if "history" not in st.session_state:
        st.session_state["history"] = []
    if "stats" not in st.session_state:
        st.session_state["stats"] = {"queries": 0, "total_latency": 0, "avg_latency": 0}

    render_sidebar()

    # Page navigation
    tab_query, tab_explorer, tab_templates = st.tabs(["💬 Query", "📊 Data Explorer", "📋 Templates"])

    with tab_query:
        render_query_section()
        render_history()

    with tab_explorer:
        from src.ui.explorer import render_data_explorer

        render_data_explorer()

    with tab_templates:
        _render_templates_tab()

    # Footer
    st.divider()
    st.caption("🏦 LangGraph Financial Swarm v0.9.0 · Local-first multi-agent financial analysis")


def _render_templates_tab():
    """Render the query templates tab."""
    st.header("📋 Query Templates")

    from src.core.templates import TEMPLATE_CATEGORIES

    for cat, templates in TEMPLATE_CATEGORIES.items():
        st.subheader(cat.replace("_", " ").title())
        for t in templates:
            diff_color = {"easy": "🟢", "medium": "🟡", "hard": "🔴"}.get(t.difficulty, "⚪")
            with st.expander(f"{diff_color} {t.name} — {t.description}"):
                st.code(t.template, language="text")
                st.caption(f"Parameters: {', '.join(t.params)}")

                # Quick fill
                params = {}
                cols = st.columns(len(t.params))
                for i, p in enumerate(t.params):
                    params[p] = cols[i].text_input(p, key=f"tmpl_{t.name}_{p}")

                if st.button(f"Run {t.name}", key=f"run_{t.name}"):
                    if all(params.values()):
                        rendered = t.render(**params)
                        st.info(f"Query: {rendered}")
                        _execute_and_display(rendered)
                    else:
                        st.warning("Please fill all parameters.")


if __name__ == "__main__":
    main()
