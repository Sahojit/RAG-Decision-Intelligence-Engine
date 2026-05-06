import itertools
import time
from dataclasses import dataclass
from typing import Optional
from rag_decision_engine.config import settings
from rag_decision_engine.config.logging_config import get_logger
from rag_decision_engine.services.reliability_service import ScoredEvidence
logger = get_logger(__name__)
@dataclass
class ContradictionPair:
    chunk_id_a: str
    chunk_id_b: str
    text_a: str
    text_b: str
    contradiction_score: float
@dataclass
class ContradictionReport:
    detected: bool
    pairs: list[ContradictionPair]
    max_contradiction_score: float
    @property
    def pair_count(self) -> int:
        return len(self.pairs)
class ContradictionDetector:
    def __init__(
        self,
        model_name: str | None = None,
        threshold: float | None = None,
        max_pairs: int = 30,
    ) -> None:
        self._model_name = model_name or settings.nli_model
        self._threshold = threshold or settings.contradiction_threshold
        self._max_pairs = max_pairs
        self._model: Optional[object] = None
    def detect(self, evidence: list[ScoredEvidence]) -> ContradictionReport:
        if len(evidence) < 2:
            return ContradictionReport(detected=False, pairs=[], max_contradiction_score=0.0)
        if settings.disable_contradiction:
            logger.info("contradiction_disabled")
            return ContradictionReport(detected=False, pairs=[], max_contradiction_score=0.0)
        t0 = time.perf_counter()
        pairs_to_check = list(itertools.combinations(evidence, 2))[: self._max_pairs]
        model = self._get_model()
        hypotheses = [(e_a.text[:512], e_b.text[:512]) for e_a, e_b in pairs_to_check]
        scores = model.predict(hypotheses, show_progress_bar=False)
        contradiction_pairs: list[ContradictionPair] = []
        max_score = 0.0
        for (e_a, e_b), score_vec in zip(pairs_to_check, scores):
            contradiction_prob = float(self._extract_contradiction_prob(score_vec))
            max_score = max(max_score, contradiction_prob)
            if contradiction_prob >= self._threshold:
                contradiction_pairs.append(
                    ContradictionPair(
                        chunk_id_a=e_a.chunk_id,
                        chunk_id_b=e_b.chunk_id,
                        text_a=e_a.text[:300],
                        text_b=e_b.text[:300],
                        contradiction_score=round(contradiction_prob, 4),
                    )
                )
        contradiction_pairs.sort(key=lambda p: p.contradiction_score, reverse=True)
        report = ContradictionReport(
            detected=bool(contradiction_pairs),
            pairs=contradiction_pairs,
            max_contradiction_score=round(max_score, 4),
        )
        logger.info(
            "contradiction_detection",
            pairs_checked=len(pairs_to_check),
            contradictions_found=len(contradiction_pairs),
            max_score=round(max_score, 3),
            elapsed_ms=round((time.perf_counter() - t0) * 1000, 1),
        )
        return report
    @staticmethod
    def _extract_contradiction_prob(score_vec: object) -> float:
        import numpy as np
        arr = np.array(score_vec, dtype=np.float32)
        if arr.ndim == 0:
            return float(arr)
        exp = np.exp(arr - arr.max())
        probs = exp / exp.sum()
        return float(probs[0])
    def _get_model(self) -> object:
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:
                raise ImportError("Install sentence-transformers") from exc
            logger.info("loading_nli_model", model=self._model_name)
            self._model = CrossEncoder(self._model_name, num_labels=3)
        return self._model
