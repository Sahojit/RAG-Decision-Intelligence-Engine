import os
import re
import time
from typing import Any
import requests
import streamlit as st
st.set_page_config(
    page_title="Evidentia — Decision Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown("""
<style>
.decision-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border: 1px solid #0f3460;
    border-radius: 12px;
    padding: 24px 28px;
    margin-bottom: 16px;
}
.verdict-win {
    font-size: 2.2rem;
    font-weight: 800;
    color: #00d4aa;
    letter-spacing: -0.5px;
}
.verdict-inconclusive {
    font-size: 1.6rem;
    font-weight: 700;
    color: #f4a261;
}
.quality-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.82rem;
    font-weight: 600;
}
.badge-high   { background: #003d2e; color: #00d4aa; border: 1px solid #00d4aa44; }
.badge-medium { background: #2d2000; color: #f4a261; border: 1px solid #f4a26144; }
.badge-low    { background: #2d0a0a; color: #e07070; border: 1px solid #e0707044; }
.section-label {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #888;
    margin-bottom: 4px;
}
div[data-testid="stMetricValue"] { font-size: 1.5rem !important; font-weight: 700 !important; }
</style>
""", unsafe_allow_html=True)
if "search_history" not in st.session_state:
    st.session_state.search_history = []
if "last_report" not in st.session_state:
    st.session_state.last_report = None
_DEFAULT_URL = os.environ.get("API_URL", "http://localhost:8080")
EXAMPLE_QUERIES = [
    "Should I use XGBoost or Random Forest for tabular datasets?",
    "Is PyTorch or TensorFlow better for production ML deployment?",
    "Should I use PostgreSQL or MongoDB for a high-write workload?",
    "FastAPI or Flask for building a microservice API?",
]
_COMPARATIVE_RE = re.compile(
    r"\bor\b|\bvs\.?\b|\bversus\b|\bcompare\b|\bbetter\b|\bbest\b|\bover\b|\bprefer\b",
    re.I,
)
def has_comparative_intent(query: str) -> bool:
    return bool(_COMPARATIVE_RE.search(query))
def api_call(method: str, path: str, **kwargs) -> dict | None:
    try:
        url = st.session_state.get("api_url", _DEFAULT_URL).rstrip("/") + path
        resp = getattr(requests, method)(url, **kwargs)
        resp.raise_for_status()
        return resp.json()
    except requests.ConnectionError:
        st.error("Cannot reach the API. Make sure the backend is running.")
    except requests.HTTPError as e:
        st.error(f"API returned an error: {e.response.status_code}")
    except requests.Timeout:
        st.error("Request timed out. The pipeline may still be processing — try again.")
    except Exception as exc:
        st.error(f"Unexpected error: {exc}")
    return None
def get_health() -> bool:
    try:
        r = requests.get(
            st.session_state.get("api_url", _DEFAULT_URL).rstrip("/") + "/health",
            timeout=3,
        )
        return r.status_code == 200
    except Exception:
        return False
def derive_quality(report: dict) -> dict:
    options = report.get("options", [])
    avg_cred = (
        sum(o.get("source_credibility", 0) for o in options) / len(options)
        if options else 0
    )
    total_docs = sum(o.get("supporting_documents", 0) for o in options)
    contradictions = report.get("contradiction_count", 0)
    if avg_cred >= 0.75:
        source_label, source_cls = "High", "badge-high"
    elif avg_cred >= 0.55:
        source_label, source_cls = "Medium", "badge-medium"
    else:
        source_label, source_cls = "Low", "badge-low"
    if contradictions == 0:
        consistency_label, consistency_cls = "Good", "badge-high"
    elif contradictions <= 5:
        consistency_label, consistency_cls = "Mixed", "badge-medium"
    else:
        consistency_label, consistency_cls = "Conflicting", "badge-low"
    if total_docs >= 12:
        coverage_label, coverage_cls = "Sufficient", "badge-high"
    elif total_docs >= 5:
        coverage_label, coverage_cls = "Moderate", "badge-medium"
    else:
        coverage_label, coverage_cls = "Limited", "badge-low"
    return {
        "source": (source_label, source_cls),
        "consistency": (consistency_label, consistency_cls),
        "coverage": (coverage_label, coverage_cls),
        "avg_cred": avg_cred,
        "total_docs": total_docs,
    }
def badge_html(label: str, css_class: str) -> str:
    return f'<span class="quality-badge {css_class}">{label}</span>'
def render_sidebar():
    with st.sidebar:
        st.markdown("## ⚡ Evidentia")
        st.caption("RAG Decision Intelligence Engine")
        st.divider()
        api_url = st.text_input(
            "API endpoint",
            value=_DEFAULT_URL,
            label_visibility="collapsed",
            placeholder="http://localhost:8080",
        )
        st.session_state["api_url"] = api_url
        is_online = get_health()
        if is_online:
            st.success("API Online", icon="🟢")
        else:
            st.error("API Offline", icon="🔴")
        st.divider()
        st.markdown("##### Search History")
        history = st.session_state.search_history
        if not history:
            st.caption("Your queries will appear here.")
        else:
            if st.button("Clear", use_container_width=True, type="secondary"):
                st.session_state.search_history = []
                st.rerun()
            for i, entry in enumerate(reversed(history[-10:])):
                rec = entry.get("recommendation") or "Inconclusive"
                icon = "🟢" if entry.get("recommendation") else "🟡"
                conf = round(entry.get("confidence", 0) * 100, 1)
                with st.expander(f"{icon} {entry['query'][:36]}…", expanded=False):
                    st.caption(f"**{rec}** · {conf}% · {entry['ts']}")
                    if st.button("Re-run", key=f"rerun_{i}_{entry['ts']}", use_container_width=True):
                        st.session_state["prefill_query"] = entry["query"]
                        st.rerun()
        st.divider()
        with st.expander("About this system"):
            st.markdown(
                "**Evidentia** retrieves evidence from your knowledge base, "
                "scores reliability with XGBoost, detects contradictions via NLI, "
                "applies a decision policy engine, and generates a structured recommendation."
            )
            st.markdown("**Stack:** FAISS · BM25 · CrossEncoder · XGBoost · Ollama")
def render_query_panel() -> tuple[str, bool, bool]:
    col_title, col_status = st.columns([6, 1])
    with col_title:
        st.markdown("## Ask a Decision Question")
    with col_status:
        st.markdown("")
    st.markdown(
        "<p style='color:#888; margin-top:-12px;'>Compare two options — the engine finds evidence, scores it, and gives you a clear recommendation.</p>",
        unsafe_allow_html=True,
    )
    with st.expander("Examples"):
        cols = st.columns(2)
        for idx, q in enumerate(EXAMPLE_QUERIES):
            if cols[idx % 2].button(q, key=f"eg_{idx}", use_container_width=True):
                st.session_state["prefill_query"] = q
                st.rerun()
    query = st.text_area(
        "Your question",
        value=st.session_state.pop("prefill_query", st.session_state.get("prefill_query", "")),
        height=90,
        placeholder="e.g. Should I use XGBoost or Random Forest for tabular datasets?",
        label_visibility="collapsed",
    )
    if query.strip() and not has_comparative_intent(query):
        st.info(
            "💡 This system works best for comparison queries — try phrasing as **X vs Y** or **X or Y**.",
            icon="💡",
        )
    with st.expander("⚙️ Retrieval Settings", expanded=False):
        fc1, fc2, fc3 = st.columns([2, 2, 2])
        with fc1:
            source_type = st.selectbox(
                "Source type",
                ["All sources", "Research papers", "Documentation", "Blog posts"],
            )
        with fc2:
            min_year = st.slider("Published after", 2015, 2026, 2018)
        with fc3:
            st.markdown("")
            st.markdown("")
            cited_only = st.toggle("Cited sources only", value=False)
        use_live = st.toggle(
            "Fetch live research (arXiv + Semantic Scholar)",
            value=False,
            help="Queries external APIs for fresh papers. Adds ~10–30s latency.",
        )
        if use_live:
            st.caption("Live retrieval fetches up to 10 papers from arXiv and Semantic Scholar in real time.")
    filter_suffix_parts = []
    if source_type != "All sources":
        label_map = {
            "Research papers": "research papers",
            "Documentation": "documentation",
            "Blog posts": "blog posts",
        }
        filter_suffix_parts.append(f"only {label_map[source_type]}")
    if min_year > 2015:
        filter_suffix_parts.append(f"after {min_year}")
    if cited_only:
        filter_suffix_parts.append("with citations")
    augmented_query = query.strip()
    if filter_suffix_parts and augmented_query:
        augmented_query = f"{augmented_query}. Use {', '.join(filter_suffix_parts)}."
    run = st.button("Analyse →", type="primary", use_container_width=True)
    return augmented_query, run, use_live
def render_verdict(report: dict):
    recommended = report.get("recommended_option")
    confidence = report.get("recommendation_confidence", 0.0)
    conf_pct = round(confidence * 100, 1)
    reasoning = report.get("reasoning", "")
    short_reasoning = ". ".join(reasoning.split(". ")[:3]).strip()
    if not short_reasoning.endswith("."):
        short_reasoning += "."
    if recommended:
        st.markdown(
            f'<div class="decision-card">'
            f'<div class="section-label">Recommendation</div>'
            f'<div class="verdict-win">{recommended}</div>'
            f'<div style="color:#aaa; font-size:1rem; margin-top:6px;">Confidence: <strong style="color:#00d4aa">{conf_pct}%</strong></div>'
            f'<hr style="border-color:#0f3460; margin: 14px 0;">'
            f'<div style="color:#ccc; font-size:0.95rem; line-height:1.6">{short_reasoning}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        policy_reason = report.get("policy_reason", "Evidence is conflicting or insufficient.")
        st.markdown(
            f'<div class="decision-card">'
            f'<div class="section-label">Result</div>'
            f'<div class="verdict-inconclusive">No clear recommendation</div>'
            f'<div style="color:#aaa; font-size:0.95rem; margin-top:10px; line-height:1.6">{policy_reason}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
def render_status_panel(report: dict):
    q = derive_quality(report)
    live_used = report.get("live_retrieval_used", False)
    live_count = report.get("live_docs_count", 0)
    local_count = report.get("local_docs_count", 0)
    active_filters = {k: v for k, v in report.get("filters_applied", {}).items() if v is not None}
    st.markdown("##### Evidence Quality")
    badge_col1, badge_col2, badge_col3 = st.columns(3)
    with badge_col1:
        st.markdown(f"**Source Quality**")
        st.markdown(badge_html(*q["source"]), unsafe_allow_html=True)
    with badge_col2:
        st.markdown(f"**Consistency**")
        st.markdown(badge_html(*q["consistency"]), unsafe_allow_html=True)
    with badge_col3:
        st.markdown(f"**Coverage**")
        st.markdown(badge_html(*q["coverage"]), unsafe_allow_html=True)
    st.markdown("")
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Total Evidence", q["total_docs"])
    mc2.metric("Local Docs", local_count)
    mc3.metric("Live Docs", live_count, delta="arXiv · S2" if live_count else None)
    mc4.metric("Filters", len(active_filters))
    if active_filters:
        filter_text = " · ".join(f"{k}: {v}" for k, v in active_filters.items())
        st.caption(f"Active filters: {filter_text}")
    if report.get("reflection_flag"):
        st.warning(
            f"**Quality Warning** — {report.get('reflection_reason', '')}",
            icon="⚠️",
        )
def render_contradiction_banner(report: dict):
    n = report.get("contradiction_count", 0)
    detected = report.get("contradiction_detected", False)
    if not detected:
        st.success("Evidence is internally consistent — no contradictions found.", icon="✅")
        return
    st.warning(
        f"**Evidence inconsistency detected** — {n} conflicting signal{'s' if n != 1 else ''} found in retrieved documents.",
        icon="⚠️",
    )
    if report.get("contradiction_details"):
        with st.expander("View conflicting evidence"):
            for pair in report["contradiction_details"][:3]:
                col_a, col_b = st.columns(2)
                col_a.caption("Signal A")
                col_a.info(pair.get("snippet_a", "")[:200])
                col_b.caption("Signal B")
                col_b.error(pair.get("snippet_b", "")[:200])
                st.caption(f"Conflict score: {pair.get('score', 0):.2f}")
                st.divider()
def render_advanced_section(report: dict):
    with st.expander("🔍 Detailed Analysis"):
        recommended = report.get("recommended_option")
        st.markdown("##### Evidence by Option")
        for opt in report.get("options", []):
            is_winner = opt["option"] == recommended
            label = f"{'✅ ' if is_winner else ''}{opt['option']}"
            with st.expander(label, expanded=is_winner):
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Docs", opt["supporting_documents"])
                c2.metric("Reliability", f"{opt['reliability_score']:.2f}")
                c3.metric("Credibility", f"{opt['source_credibility']:.2f}")
                c4.metric("Evidence Score", f"{opt['final_evidence_score']:.2f}")
                snippets = opt.get("top_evidence_snippets", [])[:3]
                if snippets:
                    st.markdown("**Supporting evidence:**")
                    for snippet in snippets:
                        st.markdown(f"> {snippet[:250]}")
        if report.get("policy_reason"):
            st.markdown("---")
            st.markdown("##### Policy Decision")
            st.info(report["policy_reason"])
        if report.get("reflection_flag"):
            st.markdown("---")
            st.markdown("##### Reflection Agent")
            st.warning(report.get("reflection_reason", ""))
def render_debug_section(report: dict):
    with st.expander("🛠 Developer Debug"):
        st.caption("Raw pipeline output — for engineering review only.")
        dc1, dc2 = st.columns(2)
        with dc1:
            st.markdown("**Scores**")
            for opt in report.get("options", []):
                st.code(
                    f"{opt['option'][:40]}\n"
                    f"  final:       {opt['final_evidence_score']:.4f}\n"
                    f"  reliability: {opt['reliability_score']:.4f}\n"
                    f"  credibility: {opt['source_credibility']:.4f}\n"
                    f"  similarity:  {opt['avg_similarity']:.4f}\n"
                    f"  docs:        {opt['supporting_documents']}",
                    language="yaml",
                )
        with dc2:
            st.markdown("**Pipeline Metadata**")
            st.code(
                f"latency_ms:       {report.get('latency_ms')}\n"
                f"model_used:       {report.get('model_used')}\n"
                f"live_retrieval:   {report.get('live_retrieval_used')}\n"
                f"live_docs:        {report.get('live_docs_count')}\n"
                f"local_docs:       {report.get('local_docs_count')}\n"
                f"contradictions:   {report.get('contradiction_count')}\n"
                f"reflection_flag:  {report.get('reflection_flag')}",
                language="yaml",
            )
        st.markdown("**Full JSON**")
        st.json(report)
def render_ingest_tab():
    st.markdown("## Add to Knowledge Base")
    st.markdown("<p style='color:#888;'>Documents ingested here become evidence for future queries.</p>", unsafe_allow_html=True)
    mode = st.radio("Input mode", ["Paste text", "File paths"], horizontal=True)
    if mode == "Paste text":
        text = st.text_area(
            "Document content",
            height=220,
            placeholder="Paste any text — research papers, docs, articles…",
            label_visibility="collapsed",
        )
        if st.button("Ingest →", type="primary") and text.strip():
            with st.spinner("Processing…"):
                result = api_call("post", "/ingest_documents", json={"texts": [text.strip()]}, timeout=60)
            if result:
                st.success(
                    f"Added {result.get('ingested_documents', 0)} document · "
                    f"{result.get('ingested_chunks', 0)} chunks indexed."
                )
    else:
        paths_input = st.text_area(
            "File paths",
            height=140,
            placeholder="/path/to/paper.pdf\n/path/to/notes.txt",
            label_visibility="collapsed",
        )
        if st.button("Ingest Files →", type="primary") and paths_input.strip():
            paths = [p.strip() for p in paths_input.splitlines() if p.strip()]
            with st.spinner(f"Processing {len(paths)} file(s)…"):
                result = api_call("post", "/ingest_documents", json={"file_paths": paths}, timeout=120)
            if result:
                st.success(f"Ingested {result.get('ingested_documents', 0)} document(s).")
def render_metrics_tab():
    st.markdown("## System Metrics")
    if st.button("Refresh", type="secondary"):
        st.rerun()
    m = api_call("get", "/metrics", timeout=5)
    if not m:
        st.warning("Could not reach the API metrics endpoint.")
        return
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Requests", m.get("request_count", 0))
    c2.metric("Decisions", m.get("decision_count", 0))
    c3.metric("Docs Ingested", m.get("ingest_count", 0))
    c4.metric("Errors", m.get("error_count", 0))
    c5.metric("Avg Latency", f"{m.get('avg_latency_ms', 0)} ms")
render_sidebar()
tab_decision, tab_ingest, tab_metrics = st.tabs(["Decision Analysis", "Knowledge Base", "Metrics"])
with tab_decision:
    augmented_query, run_clicked, use_live = render_query_panel()
    if run_clicked:
        if not augmented_query:
            st.warning("Please enter a question before running.")
        else:
            spinner_text = "Fetching live research + analysing evidence…" if use_live else "Analysing evidence…"
            with st.spinner(spinner_text):
                t0 = time.perf_counter()
                report = api_call(
                    "post",
                    "/decision",
                    json={"query": augmented_query, "use_live_retrieval": use_live},
                    timeout=180,
                )
                elapsed_ms = round((time.perf_counter() - t0) * 1000)
            if report:
                total_docs = sum(o.get("supporting_documents", 0) for o in report.get("options", []))
                if total_docs == 0:
                    st.error("No relevant evidence found — try rephrasing your query or ingesting more documents.")
                else:
                    st.session_state.last_report = report
                    st.session_state.search_history.append({
                        "query": augmented_query,
                        "recommendation": report.get("recommended_option"),
                        "confidence": report.get("recommendation_confidence", 0.0),
                        "ts": time.strftime("%H:%M:%S"),
                    })
                    st.caption(f"Completed in {elapsed_ms:,} ms")
                    render_verdict(report)
                    st.markdown("---")
                    render_status_panel(report)
                    st.markdown("---")
                    render_contradiction_banner(report)
                    st.markdown("---")
                    render_advanced_section(report)
                    render_debug_section(report)
    elif st.session_state.last_report:
        report = st.session_state.last_report
        st.caption("Showing last result — run a new query above.")
        render_verdict(report)
        st.markdown("---")
        render_status_panel(report)
        st.markdown("---")
        render_contradiction_banner(report)
        st.markdown("---")
        render_advanced_section(report)
        render_debug_section(report)
with tab_ingest:
    render_ingest_tab()
with tab_metrics:
    render_metrics_tab()
