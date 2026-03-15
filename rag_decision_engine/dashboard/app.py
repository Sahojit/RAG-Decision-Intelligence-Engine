import json
import os
import time
from typing import Any
import requests
import streamlit as st
st.set_page_config(
    page_title="RAG Decision Intelligence Engine",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)
if "search_history" not in st.session_state:
    st.session_state.search_history = []
with st.sidebar:
    st.title("⚙️ Configuration")
    _default_url = os.environ.get("API_URL", "http://localhost:8080")
    api_url = st.text_input("API Base URL", value=_default_url)
    st.divider()
    st.markdown("### About")
    st.markdown(
        "Evidence-based decision intelligence system using "
        "**Hybrid RAG**, **NLI contradiction detection**, and "
        "**ML reliability scoring**."
    )
    st.divider()
    st.markdown("### 🕓 Search History")
    if not st.session_state.search_history:
        st.caption("No searches yet.")
    else:
        if st.button("🗑 Clear history", use_container_width=True):
            st.session_state.search_history = []
            st.rerun()
        for i, entry in enumerate(reversed(st.session_state.search_history)):
            rec = entry.get("recommendation") or "Inconclusive"
            conf = entry.get("confidence", 0.0)
            conf_pct = f"{round(conf * 100, 1)}%"
            label = f"**{entry['query'][:42]}{'…' if len(entry['query']) > 42 else ''}**"
            badge = "🟢" if entry.get("recommendation") else "🔴"
            with st.expander(f"{badge} {entry['query'][:38]}{'…' if len(entry['query']) > 38 else ''}", expanded=False):
                st.markdown(f"**Recommendation:** {rec}")
                st.markdown(f"**Confidence:** {conf_pct}")
                st.markdown(f"**Time:** {entry['ts']}")
                if st.button("↩ Re-run", key=f"rerun_{i}_{entry['ts']}"):
                    st.session_state["query_input"] = entry["query"]
                    st.rerun()
API_URL = api_url.rstrip("/")
def call_decision(query: str) -> dict[str, Any] | None:
    try:
        resp = requests.post(
            f"{API_URL}/decision",
            json={"query": query},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as exc:
        st.error(f"API error: {exc}")
        return None
def call_ingest(texts: list[str]) -> dict[str, Any] | None:
    try:
        resp = requests.post(
            f"{API_URL}/ingest_documents",
            json={"texts": texts},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as exc:
        st.error(f"Ingestion error: {exc}")
        return None
def call_metrics() -> dict[str, Any]:
    try:
        resp = requests.get(f"{API_URL}/metrics", timeout=5)
        return resp.json()
    except Exception:
        return {}
def health_badge() -> str:
    try:
        resp = requests.get(f"{API_URL}/health", timeout=3)
        if resp.status_code == 200:
            return "🟢 API Online"
    except Exception:
        pass
    return "🔴 API Offline"
tab_decision, tab_ingest, tab_metrics = st.tabs(
    ["🔍 Decision Analysis", "📥 Ingest Documents", "📊 Metrics"]
)
with tab_decision:
    col_header, col_status = st.columns([5, 1])
    with col_header:
        st.title("🧠 Decision Intelligence Engine")
    with col_status:
        st.markdown(f"**{health_badge()}**")
    st.markdown(
        "Ask a comparative or strategic question. The engine retrieves evidence, "
        "scores reliability, detects contradictions, and generates a structured report."
    )
    example_queries = [
        "Should I use XGBoost or Random Forest for tabular datasets?",
        "Is PyTorch or TensorFlow better for production deployment?",
        "Should I use PostgreSQL or MongoDB for a high-write workload?",
    ]
    with st.expander("💡 Example queries"):
        for q in example_queries:
            if st.button(q, key=q):
                st.session_state["query_input"] = q
    query = st.text_area(
        "Enter your decision query",
        value=st.session_state.get("query_input", ""),
        height=80,
        placeholder="e.g. Should I use XGBoost or Random Forest for tabular datasets?",
    )
    run_btn = st.button("🚀 Analyse", type="primary", use_container_width=True)
    if run_btn and query.strip():
        with st.spinner("Running decision pipeline…"):
            t0 = time.perf_counter()
            report = call_decision(query.strip())
            elapsed = round((time.perf_counter() - t0) * 1000)
        if report:
            st.session_state.search_history.append({
                "query": query.strip(),
                "recommendation": report.get("recommended_option"),
                "confidence": report.get("recommendation_confidence", 0.0),
                "ts": time.strftime("%H:%M:%S"),
            })
            st.success(f"Report generated in **{elapsed} ms**")
            st.divider()
            recommended = report.get("recommended_option", "N/A")
            confidence = report.get("recommendation_confidence", 0.0)
            col_rec, col_conf = st.columns(2)
            with col_rec:
                rec_label = recommended if recommended else "Inconclusive"
                st.metric("✅ Recommended Option", rec_label)
            with col_conf:
                conf_pct = round(confidence * 100, 1)
                st.metric("📈 Adjusted Confidence", f"{conf_pct}%")
            policy_reason = report.get("policy_reason", "")
            if policy_reason:
                if not recommended:
                    st.error(f"🚫 **Policy Decision:** {policy_reason}")
                else:
                    st.info(f"📋 **Policy Reasoning:** {policy_reason}")
            if report.get("reflection_flag"):
                reflection_reason = report.get("reflection_reason", "")
                st.warning(
                    f"⚠ **Reflection Warning:** {reflection_reason}",
                    icon="⚠️",
                )
            if report.get("contradiction_detected"):
                n = report.get("contradiction_count", 0)
                st.warning(
                    f"⚠️ **Contradictory evidence detected** — {n} contradicting pair(s). "
                    "Review the details below before making a decision.",
                    icon="⚠️",
                )
            else:
                st.info("✅ No contradictory evidence detected.", icon="✅")
            st.divider()
            st.subheader("📋 Evidence by Option")
            options_data = report.get("options", [])
            for opt in options_data:
                with st.expander(
                    f"**{opt['option']}**  —  "
                    f"Score: {opt['final_evidence_score']:.3f}  |  "
                    f"Docs: {opt['supporting_documents']}",
                    expanded=(opt["option"] == recommended),
                ):
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Supporting Docs", opt["supporting_documents"])
                    c2.metric("Reliability", f"{opt['reliability_score']:.3f}")
                    c3.metric("Credibility", f"{opt['source_credibility']:.3f}")
                    c4.metric("Evidence Score", f"{opt['final_evidence_score']:.3f}")
                    st.markdown("**Top Evidence Snippets**")
                    for i, snippet in enumerate(
                        opt.get("top_evidence_snippets", [])[:3], 1
                    ):
                        st.markdown(f"> {i}. {snippet}")
            st.divider()
            st.subheader("💬 AI Reasoning")
            st.markdown(report.get("reasoning", "No reasoning available."))
            if report.get("contradiction_details"):
                st.divider()
                st.subheader("🔴 Contradiction Details")
                for pair in report["contradiction_details"]:
                    with st.expander(
                        f"Contradiction — score: {pair.get('score', 0):.3f}"
                    ):
                        col_a, col_b = st.columns(2)
                        col_a.markdown("**Evidence A**")
                        col_a.info(pair.get("snippet_a", ""))
                        col_b.markdown("**Evidence B**")
                        col_b.error(pair.get("snippet_b", ""))
            with st.expander("🔧 Raw JSON Report"):
                st.json(report)
    elif run_btn:
        st.warning("Please enter a query.")
with tab_ingest:
    st.title("📥 Document Ingestion")
    st.markdown(
        "Add documents to the knowledge base. Paste raw text or enter file paths."
    )
    ingest_mode = st.radio(
        "Input mode",
        ["Paste Text", "File Paths"],
        horizontal=True,
    )
    if ingest_mode == "Paste Text":
        raw_text = st.text_area(
            "Document text",
            height=250,
            placeholder="Paste document content here…",
        )
        ingest_btn = st.button("Ingest Text", type="primary")
        if ingest_btn and raw_text.strip():
            with st.spinner("Ingesting…"):
                result = call_ingest([raw_text.strip()])
            if result:
                st.success(
                    f"Ingested {result.get('ingested_documents', 0)} document(s), "
                    f"{result.get('ingested_chunks', 0)} chunks."
                )
                st.json(result)
    else:
        file_paths_input = st.text_area(
            "File paths (one per line)",
            height=150,
            placeholder="/path/to/doc.pdf\n/path/to/doc.txt",
        )
        ingest_btn = st.button("Ingest Files", type="primary")
        if ingest_btn and file_paths_input.strip():
            paths = [p.strip() for p in file_paths_input.strip().splitlines() if p.strip()]
            with st.spinner(f"Ingesting {len(paths)} file(s)…"):
                try:
                    resp = requests.post(
                        f"{API_URL}/ingest_documents",
                        json={"file_paths": paths},
                        timeout=120,
                    )
                    resp.raise_for_status()
                    result = resp.json()
                    st.success(
                        f"Ingested {result.get('ingested_documents', 0)} document(s)."
                    )
                    st.json(result)
                except requests.RequestException as exc:
                    st.error(f"Error: {exc}")
with tab_metrics:
    st.title("📊 Operational Metrics")
    refresh = st.button("🔄 Refresh Metrics")
    if refresh or True:
        m = call_metrics()
        if m:
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Total Requests", m.get("request_count", 0))
            c2.metric("Decisions", m.get("decision_count", 0))
            c3.metric("Documents Ingested", m.get("ingest_count", 0))
            c4.metric("Errors", m.get("error_count", 0))
            c5.metric("Avg Latency (ms)", m.get("avg_latency_ms", 0))
        else:
            st.warning("Could not fetch metrics. Is the API running?")
