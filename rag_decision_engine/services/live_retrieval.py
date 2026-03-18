import asyncio
import re
import xml.etree.ElementTree as ET
from rag_decision_engine.config.logging_config import get_logger
from rag_decision_engine.retrieval.vector_retriever import RetrievedDocument
logger = get_logger(__name__)
_ARXIV_URL = "https://export.arxiv.org/api/query"
_SEMANTIC_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
_TIMEOUT = 8.0
_MAX_RESULTS = 5
_ML_TERMS = re.compile(
    r"\b(machine learning|deep learning|neural network|transformer|llm|gpt|bert|"
    r"xgboost|random forest|faiss|embedding|retrieval|rag|nlp|"
    r"classification|regression|clustering|reinforcement|attention|"
    r"pytorch|tensorflow|scikit|model|algorithm|dataset|benchmark)\b",
    re.I,
)
_RESEARCH_TERMS = re.compile(
    r"\b(research|paper|study|survey|literature|publication|findings|experiment)\b",
    re.I,
)
def detect_query_type(query: str) -> bool:
    return bool(_ML_TERMS.search(query) or _RESEARCH_TERMS.search(query))
async def fetch_arxiv(query: str, max_results: int = _MAX_RESULTS) -> list[RetrievedDocument]:
    try:
        import httpx
        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance",
        }
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.get(_ARXIV_URL, params=params)
            resp.raise_for_status()
        docs = _parse_arxiv(resp.text)
        logger.info("arxiv_fetched", count=len(docs), query_preview=query[:60])
        return docs
    except Exception as exc:
        logger.warning("arxiv_fetch_failed", error=str(exc))
        return []
def _parse_arxiv(xml_text: str) -> list[RetrievedDocument]:
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []
    docs = []
    for i, entry in enumerate(root.findall("atom:entry", ns)):
        title_el = entry.find("atom:title", ns)
        summary_el = entry.find("atom:summary", ns)
        published_el = entry.find("atom:published", ns)
        id_el = entry.find("atom:id", ns)
        title = (title_el.text or "").strip()
        summary = (summary_el.text or "").strip()
        published = (published_el.text or "")[:4]
        arxiv_id = (id_el.text or "").split("/")[-1] if id_el is not None else str(i)
        text = f"{title}\n\n{summary}".strip() if title else summary
        if not text:
            continue
        year = int(published) if published.isdigit() else None
        docs.append(
            RetrievedDocument(
                chunk_id=f"arxiv_{arxiv_id}",
                doc_id=f"arxiv_{arxiv_id}",
                text=text,
                score=0.5,
                metadata={
                    "source_type": "research_paper",
                    "estimated_year": year,
                    "year": year,
                    "has_citations": True,
                    "origin": "arxiv",
                    "title": title,
                },
                retriever="live_arxiv",
            )
        )
    return docs
async def fetch_semantic_scholar(query: str, max_results: int = _MAX_RESULTS) -> list[RetrievedDocument]:
    try:
        import httpx
        params = {
            "query": query,
            "limit": max_results,
            "fields": "title,abstract,year,citationCount",
        }
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.get(_SEMANTIC_URL, params=params)
            resp.raise_for_status()
        docs = _parse_semantic(resp.json())
        logger.info("semantic_scholar_fetched", count=len(docs), query_preview=query[:60])
        return docs
    except Exception as exc:
        logger.warning("semantic_scholar_fetch_failed", error=str(exc))
        return []
def _parse_semantic(data: dict) -> list[RetrievedDocument]:
    docs = []
    for i, paper in enumerate(data.get("data", [])):
        title = (paper.get("title") or "").strip()
        abstract = (paper.get("abstract") or "").strip()
        year = paper.get("year")
        citations = paper.get("citationCount") or 0
        paper_id = paper.get("paperId") or str(i)
        text = f"{title}\n\n{abstract}".strip() if title else abstract
        if not text:
            continue
        docs.append(
            RetrievedDocument(
                chunk_id=f"ss_{paper_id}",
                doc_id=f"ss_{paper_id}",
                text=text,
                score=0.5,
                metadata={
                    "source_type": "research_paper",
                    "estimated_year": year,
                    "year": year,
                    "has_citations": citations > 0,
                    "citation_count": citations,
                    "origin": "semantic_scholar",
                    "title": title,
                },
                retriever="live_semantic_scholar",
            )
        )
    return docs
async def fetch_live_documents(query: str) -> list[RetrievedDocument]:
    results = await asyncio.gather(
        fetch_arxiv(query),
        fetch_semantic_scholar(query),
        return_exceptions=True,
    )
    docs: list[RetrievedDocument] = []
    for r in results:
        if isinstance(r, list):
            docs.extend(r)
        elif isinstance(r, Exception):
            logger.warning("live_retrieval_partial_failure", error=str(r))
    logger.info("live_retrieval_complete", total_docs=len(docs))
    return docs
def fetch_live_documents_sync(query: str) -> list[RetrievedDocument]:
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, fetch_live_documents(query))
                return future.result(timeout=20)
        return loop.run_until_complete(fetch_live_documents(query))
    except Exception as exc:
        logger.warning("live_retrieval_sync_failed", error=str(exc))
        return []
