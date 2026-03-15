import json
import pickle
import time
from pathlib import Path
from typing import Any
import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from rag_decision_engine.config import settings
from rag_decision_engine.config.logging_config import get_logger
logger = get_logger(__name__)
FEATURE_COLUMNS = [
    "similarity_score",
    "doc_length_tokens",
    "has_citations",
    "publication_year_norm",
    "source_credibility",
    "chunk_index_norm",
]
def build_pipeline(model_type: str | None = None) -> Pipeline:
    mtype = model_type or settings.ml_model_type
    if mtype == "xgboost":
        try:
            from xgboost import XGBClassifier
            clf = XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1,
            )
        except ImportError as exc:
            raise ImportError("Install xgboost: pip install xgboost") from exc
    else:
        clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", clf),
        ]
    )
def train(
    df: pd.DataFrame,
    model_type: str | None = None,
    cv_folds: int = 5,
) -> dict[str, Any]:
    mtype = model_type or settings.ml_model_type
    X = df[FEATURE_COLUMNS].values.astype(np.float32)
    y = df["label"].values.astype(int)
    pipeline = build_pipeline(mtype)
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment)
    with mlflow.start_run(run_name=f"train_{mtype}_{int(time.time())}"):
        mlflow.log_param("model_type", mtype)
        mlflow.log_param("cv_folds", cv_folds)
        mlflow.log_param("n_samples", len(df))
        mlflow.log_param("feature_columns", FEATURE_COLUMNS)
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_scores = cross_val_score(pipeline, X, y, cv=skf, scoring="f1")
        mlflow.log_metric("cv_f1_mean", float(cv_scores.mean()))
        mlflow.log_metric("cv_f1_std", float(cv_scores.std()))
        pipeline.fit(X, y)
        y_pred = pipeline.predict(X)
        metrics = {
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred, zero_division=0),
            "recall": recall_score(y, y_pred, zero_division=0),
            "f1": f1_score(y, y_pred, zero_division=0),
            "cv_f1_mean": float(cv_scores.mean()),
            "cv_f1_std": float(cv_scores.std()),
        }
        for name, val in metrics.items():
            mlflow.log_metric(name, val)
        logger.info("training_complete", **{k: round(v, 4) for k, v in metrics.items()})
        logger.info("classification_report\n%s", classification_report(y, y_pred))
        model_path = settings.ml_model_path
        model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(model_path, "wb") as f:
            pickle.dump(pipeline, f)
        mlflow.log_artifact(str(model_path))
    return {"pipeline": pipeline, "metrics": metrics}
def generate_synthetic_training_data(n_samples: int = 2000) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    similarity = rng.uniform(0.0, 1.0, n_samples)
    doc_length = rng.integers(50, 1500, n_samples).astype(float)
    has_citations = rng.choice([0, 1], n_samples, p=[0.4, 0.6]).astype(float)
    pub_year_norm = rng.uniform(0.0, 1.0, n_samples)
    source_cred = rng.choice([0.4, 0.5, 0.7, 0.9, 0.95], n_samples)
    chunk_idx_norm = rng.uniform(0.0, 1.0, n_samples)
    score = (
        0.40 * similarity
        + 0.20 * has_citations
        + 0.20 * source_cred
        + 0.10 * pub_year_norm
        + 0.05 * (doc_length / 1500.0)
        + rng.normal(0, 0.08, n_samples)
    )
    label = (score > 0.5).astype(int)
    return pd.DataFrame(
        {
            "similarity_score": similarity,
            "doc_length_tokens": doc_length,
            "has_citations": has_citations,
            "publication_year_norm": pub_year_norm,
            "source_credibility": source_cred,
            "chunk_index_norm": chunk_idx_norm,
            "label": label,
        }
    )
if __name__ == "__main__":
    from rag_decision_engine.config.logging_config import configure_logging
    configure_logging()
    df = generate_synthetic_training_data(n_samples=3000)
    result = train(df)
    print(json.dumps({k: round(v, 4) for k, v in result["metrics"].items()}, indent=2))
