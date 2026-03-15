import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np
from rag_decision_engine.config import settings
from rag_decision_engine.config.logging_config import get_logger
from rag_decision_engine.models.train_model import FEATURE_COLUMNS
logger = get_logger(__name__)
@dataclass
class EvidenceFeatures:
    similarity_score: float
    doc_length_tokens: int
    has_citations: bool
    publication_year: Optional[int]
    source_credibility: float
    chunk_index: int
    total_chunks: int = 1
    def to_array(self) -> np.ndarray:
        pub_year_norm = self._normalise_year(self.publication_year)
        chunk_index_norm = self.chunk_index / max(self.total_chunks, 1)
        return np.array(
            [
                self.similarity_score,
                float(self.doc_length_tokens),
                float(int(self.has_citations)),
                pub_year_norm,
                self.source_credibility,
                chunk_index_norm,
            ],
            dtype=np.float32,
        )
    @staticmethod
    def _normalise_year(year: Optional[int]) -> float:
        if year is None:
            return 0.5
        return max(0.0, min(1.0, (year - 2000) / 25.0))
class ReliabilityPredictor:
    _instance: Optional["ReliabilityPredictor"] = None
    def __new__(cls) -> "ReliabilityPredictor":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._pipeline = None
        return cls._instance
    def score(self, features: EvidenceFeatures) -> float:
        pipeline = self._load_pipeline()
        if pipeline is None:
            return self._heuristic_score(features)
        X = features.to_array().reshape(1, -1)
        proba = pipeline.predict_proba(X)[0]
        return float(proba[1])
    def score_batch(self, batch: list[EvidenceFeatures]) -> list[float]:
        pipeline = self._load_pipeline()
        if pipeline is None:
            return [self._heuristic_score(f) for f in batch]
        X = np.stack([f.to_array() for f in batch])
        probas = pipeline.predict_proba(X)[:, 1]
        return probas.tolist()
    def _load_pipeline(self) -> Optional[object]:
        if self._pipeline is not None:
            return self._pipeline
        model_path: Path = settings.ml_model_path
        if not model_path.exists():
            logger.warning(
                "reliability_model_not_found",
                path=str(model_path),
                fallback="heuristic",
            )
            return None
        with open(model_path, "rb") as f:
            self._pipeline = pickle.load(f)
        logger.info("reliability_model_loaded", path=str(model_path))
        return self._pipeline
    @staticmethod
    def _heuristic_score(features: EvidenceFeatures) -> float:
        return (
            0.50 * features.similarity_score
            + 0.20 * float(features.has_citations)
            + 0.20 * features.source_credibility
            + 0.10 * EvidenceFeatures._normalise_year(features.publication_year)
        )
