import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from rag_decision_engine.config.logging_config import configure_logging, get_logger
from rag_decision_engine.retrieval.hybrid_retriever import HybridRetriever
from rag_decision_engine.services.decision_service import DecisionService
logger = get_logger(__name__)
@dataclass
class EvalSample:
    query: str
    relevant_doc_ids: list[str]
    expected_decision: Optional[str]
@dataclass
class RetrievalMetrics:
    recall_at_k: float
    precision_at_k: float
    mrr: float
    k: int
@dataclass
class DecisionMetrics:
    recommendation_accuracy: float
    avg_confidence: float
    contradiction_rate: float
    avg_latency_ms: float
@dataclass
class EvaluationResult:
    retrieval: RetrievalMetrics
    decision: DecisionMetrics
    n_samples: int
    elapsed_seconds: float
    per_query_results: list[dict] = field(default_factory=list)
class Evaluator:
    def __init__(self, retrieval_k: int = 10) -> None:
        self._retriever = HybridRetriever()
        self._decision_service = DecisionService()
        self._k = retrieval_k
    def evaluate(self, samples: list[EvalSample]) -> EvaluationResult:
        t_start = time.perf_counter()
        per_query: list[dict] = []
        recalls, precisions, rrs = [], [], []
        correct_decisions = 0
        confidences = []
        contradictions = 0
        latencies = []
        for sample in samples:
            logger.info("eval_query", query=sample.query[:80])
            retrieved = self._retriever.search(sample.query, top_k=self._k)
            retrieved_ids = {doc.doc_id for doc in retrieved}
            relevant_ids = set(sample.relevant_doc_ids)
            recall = (
                len(relevant_ids & retrieved_ids) / len(relevant_ids)
                if relevant_ids else 0.0
            )
            precision = (
                len(relevant_ids & retrieved_ids) / len(retrieved_ids)
                if retrieved_ids else 0.0
            )
            rr = self._reciprocal_rank(
                [doc.doc_id for doc in retrieved], sample.relevant_doc_ids
            )
            recalls.append(recall)
            precisions.append(precision)
            rrs.append(rr)
            t0 = time.perf_counter()
            report = self._decision_service.decide(sample.query)
            lat = (time.perf_counter() - t0) * 1000
            latencies.append(lat)
            confidences.append(report.recommendation_confidence)
            if report.contradiction_detected:
                contradictions += 1
            decision_correct = False
            if sample.expected_decision and report.recommended_option:
                decision_correct = (
                    sample.expected_decision.lower()
                    in report.recommended_option.lower()
                )
            if decision_correct:
                correct_decisions += 1
            per_query.append(
                {
                    "query": sample.query,
                    "recall": round(recall, 4),
                    "precision": round(precision, 4),
                    "mrr": round(rr, 4),
                    "recommended": report.recommended_option,
                    "expected": sample.expected_decision,
                    "correct": decision_correct,
                    "confidence": round(report.recommendation_confidence, 4),
                    "contradiction": report.contradiction_detected,
                    "latency_ms": round(lat, 1),
                }
            )
        n = len(samples) or 1
        return EvaluationResult(
            retrieval=RetrievalMetrics(
                recall_at_k=round(statistics.mean(recalls), 4),
                precision_at_k=round(statistics.mean(precisions), 4),
                mrr=round(statistics.mean(rrs), 4),
                k=self._k,
            ),
            decision=DecisionMetrics(
                recommendation_accuracy=round(correct_decisions / n, 4),
                avg_confidence=round(statistics.mean(confidences), 4),
                contradiction_rate=round(contradictions / n, 4),
                avg_latency_ms=round(statistics.mean(latencies), 1),
            ),
            n_samples=n,
            elapsed_seconds=round(time.perf_counter() - t_start, 2),
            per_query_results=per_query,
        )
    @staticmethod
    def _reciprocal_rank(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
        for i, doc_id in enumerate(retrieved_ids, 1):
            if doc_id in relevant_ids:
                return 1.0 / i
        return 0.0
def build_synthetic_eval_set() -> list[EvalSample]:
    return [
        EvalSample(
            query="Should I use XGBoost or Random Forest for tabular classification?",
            relevant_doc_ids=[],
            expected_decision="XGBoost",
        ),
        EvalSample(
            query="Is PyTorch or TensorFlow better for NLP research?",
            relevant_doc_ids=[],
            expected_decision="PyTorch",
        ),
        EvalSample(
            query="Should I use PostgreSQL or MongoDB for a high-write workload?",
            relevant_doc_ids=[],
            expected_decision="PostgreSQL",
        ),
    ]
def load_eval_set(path: str | Path) -> list[EvalSample]:
    data = json.loads(Path(path).read_text())
    return [
        EvalSample(
            query=item["query"],
            relevant_doc_ids=item.get("relevant_doc_ids", []),
            expected_decision=item.get("expected_decision"),
        )
        for item in data
    ]
if __name__ == "__main__":
    configure_logging()
    evaluator = Evaluator(retrieval_k=10)
    samples = build_synthetic_eval_set()
    logger.info("evaluation_start", n_samples=len(samples))
    result = evaluator.evaluate(samples)
    print("\n" + "=" * 60)
    print(f"Evaluation Results  ({result.n_samples} samples)")
    print("=" * 60)
    print(f"Retrieval  Recall@{result.retrieval.k}:   {result.retrieval.recall_at_k:.4f}")
    print(f"Retrieval  Precision@{result.retrieval.k}: {result.retrieval.precision_at_k:.4f}")
    print(f"Retrieval  MRR:             {result.retrieval.mrr:.4f}")
    print(f"Decision   Accuracy:        {result.decision.recommendation_accuracy:.4f}")
    print(f"Decision   Avg Confidence:  {result.decision.avg_confidence:.4f}")
    print(f"Decision   Contradiction %: {result.decision.contradiction_rate:.2%}")
    print(f"Decision   Avg Latency ms:  {result.decision.avg_latency_ms:.1f}")
    print(f"Total elapsed:              {result.elapsed_seconds:.1f}s")
    print("=" * 60)
    per_q_path = Path("evaluation_results.json")
    per_q_path.write_text(json.dumps(result.per_query_results, indent=2))
    print(f"\nPer-query results written to {per_q_path}")
