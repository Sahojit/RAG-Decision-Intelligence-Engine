import os
import re
import requests
import streamlit as st
API_URL = os.environ.get("API_URL", "http://localhost:8080")
st.set_page_config(
    page_title="Evidentia — Tech Decision Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #0f1117; }
[data-testid="stSidebar"] { background: #1a1d27; border-right: 1px solid #2d2d3d; }
.verdict-card {
    background: linear-gradient(135deg, #1e2235 0%, #252840 100%);
    border: 1px solid #3d4166;
    border-radius: 12px;
    padding: 28px 32px;
    margin-bottom: 20px;
}
.verdict-title { font-size: 13px; color: #8892b0; text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 8px; }
.verdict-value { font-size: 34px; font-weight: 700; color: #e6edf3; margin-bottom: 4px; }
.badge-strong { background: #1a3a2a; color: #3fb950; border: 1px solid #3fb950; padding: 4px 12px; border-radius: 20px; font-size: 12px; font-weight: 600; }
.badge-weak { background: #3a3010; color: #d29922; border: 1px solid #d29922; padding: 4px 12px; border-radius: 20px; font-size: 12px; font-weight: 600; }
.badge-inconclusive { background: #3a1a1a; color: #f85149; border: 1px solid #f85149; padding: 4px 12px; border-radius: 20px; font-size: 12px; font-weight: 600; }
.quality-row { display: flex; gap: 12px; margin-top: 12px; flex-wrap: wrap; }
.quality-item { background: #1e2235; border: 1px solid #2d3250; border-radius: 8px; padding: 12px 16px; flex: 1; min-width: 130px; }
.quality-label { font-size: 11px; color: #8892b0; text-transform: uppercase; letter-spacing: 1px; }
.quality-value { font-size: 16px; font-weight: 600; margin-top: 4px; }
.q-high { color: #3fb950; }
.q-medium { color: #d29922; }
.q-low { color: #f85149; }
.factor-item { background: #1e2235; border-left: 3px solid #3d4166; padding: 8px 14px; border-radius: 0 6px 6px 0; margin-bottom: 6px; font-size: 14px; color: #c9d1d9; }
.snippet-box { background: #161b22; border: 1px solid #21262d; border-radius: 6px; padding: 10px 14px; margin-bottom: 8px; font-size: 13px; color: #8892b0; line-height: 1.5; }
.source-chip { display: inline-block; background: #21262d; border: 1px solid #30363d; border-radius: 4px; padding: 2px 8px; font-size: 11px; color: #8892b0; margin: 2px; }
</style>
""", unsafe_allow_html=True)
EXAMPLE_QUERIES = [
    "Should I use XGBoost or Random Forest for tabular datasets?",
    "Is PyTorch or TensorFlow better for production deployment?",
    "Should I use PostgreSQL or MongoDB for a high-write workload?",
    "FastAPI vs Flask vs Django — which for microservices?",
    "Should I use FAISS or Pinecone for vector search?",
]
def check_api() -> bool:
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False
def check_ready() -> bool:
    try:
        r = requests.get(f"{API_URL}/ready", timeout=3)
        return r.status_code == 200
    except Exception:
        return False
def call_decision(query: str, use_live: bool) -> dict:
    r = requests.post(
        f"{API_URL}/decision",
        json={"query": query, "use_live_retrieval": use_live},
        timeout=120,
    )
    r.raise_for_status()
    return r.json()
def call_metrics() -> dict:
    r = requests.get(f"{API_URL}/metrics", timeout=5)
    r.raise_for_status()
    return r.json()
def has_comparison(query: str) -> bool:
    return bool(re.search(r"\bor\b|\bvs\.?\b|\bversus\b", query, re.I))
def render_verdict(report: dict) -> None:
    rec = report.get("recommendation")
    dtype = report.get("decision_type", "inconclusive")
    conf = report.get("confidence", 0.0)
    badge_map = {
        "strong": '<span class="badge-strong">⚡ Strong Signal</span>',
        "weak": '<span class="badge-weak">⚠ Weak Signal</span>',
        "inconclusive": '<span class="badge-inconclusive">✕ Inconclusive</span>',
    }
    badge = badge_map.get(dtype, badge_map["inconclusive"])
    if dtype == "inconclusive" or not rec:
        st.markdown(f"""
<div class="verdict-card">
<div class="verdict-title">Decision</div>
<div class="verdict-value" style="color:#f85149;">No Clear Winner</div>
{badge}
<p style="color:#8892b0;margin-top:12px;font-size:14px;">Evidence is conflicting or insufficient to make a confident recommendation.</p>
</div>
""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
<div class="verdict-card">
<div class="verdict-title">Recommended</div>
<div class="verdict-value">{rec}</div>
{badge}&nbsp;&nbsp;<span style="color:#8892b0;font-size:14px;">Confidence: <strong style="color:#e6edf3;">{conf:.0%}</strong></span>
</div>
""", unsafe_allow_html=True)
def render_quality_panel(report: dict) -> None:
    options = report.get("options", [])
    contradictions = report.get("contradictions", 0)
    total_docs = sum(o.get("supporting_docs", 0) for o in options)
    avg_cred = (
        sum(o.get("credibility", 0) for o in options) / len(options)
        if options else 0.0
    )
    source_q = "High" if avg_cred >= 0.75 else ("Medium" if avg_cred >= 0.55 else "Low")
    source_cls = "q-high" if source_q == "High" else ("q-medium" if source_q == "Medium" else "q-low")
    consistency = "Conflicting" if contradictions > 3 else ("Fair" if contradictions > 0 else "Good")
    cons_cls = "q-low" if consistency == "Conflicting" else ("q-medium" if consistency == "Fair" else "q-high")
    coverage = "Sufficient" if total_docs >= 10 else ("Limited" if total_docs >= 4 else "Sparse")
    cov_cls = "q-high" if coverage == "Sufficient" else ("q-medium" if coverage == "Limited" else "q-low")
    st.markdown(f"""
<div class="quality-row">
  <div class="quality-item">
    <div class="quality-label">Source Quality</div>
    <div class="quality-value {source_cls}">{source_q}</div>
  </div>
  <div class="quality-item">
    <div class="quality-label">Consistency</div>
    <div class="quality-value {cons_cls}">{consistency}</div>
  </div>
  <div class="quality-item">
    <div class="quality-label">Coverage</div>
    <div class="quality-value {cov_cls}">{coverage} ({total_docs} docs)</div>
  </div>
</div>
""", unsafe_allow_html=True)
def render_reasoning(report: dict) -> None:
    reasoning = report.get("reasoning", "")
    if reasoning:
        st.markdown("**Analysis**")
        st.markdown(f'<div style="color:#c9d1d9;font-size:14px;line-height:1.7;padding:12px 0;">{reasoning}</div>', unsafe_allow_html=True)
def render_key_factors(report: dict) -> None:
    factors = report.get("key_factors", [])
    if not factors:
        return
    st.markdown("**Key Factors**")
    for f in factors:
        st.markdown(f'<div class="factor-item">→ {f}</div>', unsafe_allow_html=True)
def render_contradiction_banner(report: dict) -> None:
    n = report.get("contradictions", 0)
    if n == 0:
        return
    if n > 3:
        st.warning(f"⚠ Evidence inconsistency detected — {n} conflicting signals found")
    else:
        st.info(f"ℹ {n} minor evidence conflict(s) detected — confidence adjusted")
def render_evidence_attribution(report: dict) -> None:
    evidence_list = report.get("evidence", [])
    if not evidence_list:
        return
    st.markdown("**Evidence Attribution**")
    for oe in evidence_list:
        option_name = oe.get("option", "")
        items = oe.get("items", [])
        if not items:
            continue
        with st.expander(f"Evidence for: {option_name} ({len(items)} quotes)"):
            for item in items:
                quote = item.get("quote", "")
                source_type = item.get("source_type", "unknown")
                year = item.get("year")
                score = item.get("score", 0.0)
                origin = item.get("source_origin", "local")
                citations = item.get("citations", 0)
                year_str = str(year) if year else "n/a"
                origin_badge = "🌐 live" if origin == "api" else "💾 local"
                cit_badge = f'<span class="source-chip">Citations: {citations}</span>' if citations else ""
                badge_html = (
                    f'<span class="source-chip">{source_type}</span>'
                    f'<span class="source-chip">{year_str}</span>'
                    f'{cit_badge}'
                    f'<span class="source-chip">score: {score:.2f}</span>'
                    f'<span class="source-chip">{origin_badge}</span>'
                )
                st.markdown(
                    f'<div class="snippet-box">'
                    f'<div style="color:#c9d1d9;font-size:13px;line-height:1.6;border-left:3px solid #3d4166;padding-left:10px;margin-bottom:6px;">{quote}</div>'
                    f'{badge_html}'
                    f'</div>',
                    unsafe_allow_html=True,
                )
def render_advanced(report: dict) -> None:
    with st.expander("🔍 Detailed Evidence Analysis"):
        options = report.get("options", [])
        for opt in options:
            st.markdown(f"**{opt['name']}**")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Evidence Score", f"{opt['evidence_score']:.3f}")
            c2.metric("Reliability", f"{opt['reliability']:.3f}")
            c3.metric("Credibility", f"{opt['credibility']:.3f}")
            c4.metric("Docs", opt['supporting_docs'])
            snippets = opt.get("top_snippets", [])
            if snippets:
                with st.expander(f"Evidence snippets ({len(snippets)})"):
                    for s in snippets:
                        st.markdown(f'<div class="snippet-box">{s}</div>', unsafe_allow_html=True)
            st.markdown("---")
        ss = report.get("sources_summary", {})
        st.markdown("**Source Breakdown**")
        cols = st.columns(5)
        labels = ["Research Papers", "Documentation", "Blogs", "Forums", "Live API"]
        keys = ["research_papers", "documentation", "blogs", "forums", "live_api"]
        for col, label, key in zip(cols, labels, keys):
            col.metric(label, ss.get(key, 0))
        vr = report.get("validation_reason", "")
        if vr:
            st.markdown(f"**Confidence Validation:** {vr}")
def render_debug(report: dict) -> None:
    with st.expander("🛠 Developer Debug"):
        st.json(report)
def render_retrieval_info(report: dict) -> None:
    live = report.get("live_retrieval_used", False)
    live_n = report.get("live_docs_count", 0)
    local_n = report.get("local_docs_count", 0)
    filters = report.get("filters_applied", {})
    if live:
        st.markdown('<p style="color:#3fb950;font-size:12px;">⚡ Research mode — live API retrieval active (arXiv + Semantic Scholar)</p>', unsafe_allow_html=True)
    cols = st.columns(3)
    cols[0].metric("Local Docs", local_n)
    cols[1].metric("Live Docs", live_n, delta="Live" if live else None)
    cols[2].metric("Latency", f"{report.get('latency_ms', 0):.0f}ms")
    if filters:
        active = {k: v for k, v in filters.items() if v is not None}
        if active:
            chips = " ".join(f'<span class="source-chip">{k}: {v}</span>' for k, v in active.items())
            st.markdown(f"**Filters applied:** {chips}", unsafe_allow_html=True)
def sidebar() -> tuple[bool, str, int, int, bool]:
    with st.sidebar:
        st.markdown("## ⚡ Evidentia")
        st.markdown('<p style="color:#8892b0;font-size:13px;">AI Tech Decision Assistant</p>', unsafe_allow_html=True)
        api_ok = check_api()
        ready = check_ready() if api_ok else False
        if api_ok and ready:
            st.success("API Ready")
        elif api_ok:
            st.warning("API Loading...")
        else:
            st.error("API Offline")
        st.markdown("---")
        st.markdown("### Filters")
        source_options = {"All Sources": None, "Research Papers": "research_paper", "Documentation": "documentation"}
        source_label = st.selectbox("Source Type", list(source_options.keys()))
        source_type = source_options[source_label]
        year_range = st.slider("Publication Year", 2000, 2026, (2018, 2026))
        only_cited = st.toggle("Only cited sources")
        st.markdown("---")
        st.markdown("### Options")
        use_live = st.toggle("Use Live Research Data", value=False)
        st.markdown('<p style="color:#8892b0;font-size:11px;">Research queries auto-enable live retrieval.</p>', unsafe_allow_html=True)
        st.markdown("---")
        if "history" not in st.session_state:
            st.session_state.history = []
        if st.session_state.history:
            st.markdown("### History")
            for i, item in enumerate(reversed(st.session_state.history[-8:])):
                q = item["query"]
                label = q[:35] + "..." if len(q) > 35 else q
                if st.button(label, key=f"hist_{i}", use_container_width=True):
                    st.session_state.selected_query = q
        st.markdown("---")
        st.markdown('<p style="color:#444;font-size:11px;text-align:center;">Evidentia v2.0</p>', unsafe_allow_html=True)
    return use_live, source_type, year_range[0], year_range[1], only_cited
def build_filter_query(base_query: str, source_type: str | None, min_year: int, max_year: int, only_cited: bool) -> str:
    parts = [base_query]
    if source_type == "research_paper":
        parts.append("research papers")
    elif source_type == "documentation":
        parts.append("documentation")
    if min_year > 2000:
        parts.append(f"after {min_year}")
    if max_year < 2026:
        parts.append(f"before {max_year}")
    if only_cited:
        parts.append("with citations")
    return " ".join(parts)
def main() -> None:
    use_live, source_type, min_year, max_year, only_cited = sidebar()
    st.markdown("## Tech Decision Engine")
    st.markdown('<p style="color:#8892b0;">Ask a comparative technology question. The engine retrieves evidence, scores reliability, detects contradictions, and generates a structured recommendation.</p>', unsafe_allow_html=True)
    with st.expander("Example queries"):
        cols = st.columns(2)
        for i, q in enumerate(EXAMPLE_QUERIES):
            if cols[i % 2].button(q, key=f"ex_{i}", use_container_width=True):
                st.session_state.selected_query = q
    default_q = st.session_state.pop("selected_query", "")
    query = st.text_area(
        "Enter your decision query",
        value=default_q,
        height=80,
        placeholder="e.g. Should I use XGBoost or Random Forest for tabular data?",
    )
    run_col, _ = st.columns([1, 3])
    run = run_col.button("🚀 Analyse", type="primary", use_container_width=True)
    if run:
        if not query.strip():
            st.warning("Please enter a query.")
            return
        if not has_comparison(query):
            st.warning("💡 This engine works best for comparison queries — try adding 'vs', 'or', or 'versus' between two options.")
        if not check_api():
            st.error("Cannot reach API. Make sure the backend is running.")
            return
        if not check_ready():
            st.info("Models are still loading. Please wait a moment and try again.")
            return
        full_query = build_filter_query(query, source_type, min_year, max_year, only_cited)
        with st.spinner("Analysing evidence..."):
            try:
                report = call_decision(full_query, use_live)
                st.session_state.last_report = report
                st.session_state.history.append({"query": query, "result": report})
            except requests.exceptions.ConnectionError:
                st.error("Connection refused — is the API running?")
                return
            except requests.exceptions.Timeout:
                st.error("Request timed out. The pipeline may be overloaded.")
                return
            except Exception as e:
                err = str(e)
                if "No relevant" in err or "0 documents" in err:
                    st.warning("No relevant evidence found — try rephrasing your query or disabling filters.")
                else:
                    st.error(f"Pipeline error: {err}")
                return
    report = st.session_state.get("last_report")
    if not report:
        st.markdown("---")
        st.markdown('<p style="color:#8892b0;text-align:center;">Enter a query above to get started.</p>', unsafe_allow_html=True)
        return
    st.markdown("---")
    render_verdict(report)
    render_quality_panel(report)
    st.markdown("")
    render_reasoning(report)
    render_key_factors(report)
    render_contradiction_banner(report)
    st.markdown("")
    render_evidence_attribution(report)
    render_retrieval_info(report)
    render_advanced(report)
    render_debug(report)
if __name__ == "__main__":
    main()
