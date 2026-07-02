import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split


FEATURE_COLUMNS: list[str] = [
        'ratio_deuda_ingreso_ma',
        'tendencia_utilizacion',
        'volatilidad_pagos',
        'utilization_rate',
        'days_in_default',
        'num_transactions'
    ]

TARGET_COLUMN: str = 'default_indicator'

def _load_features(path: Path) -> pd.DataFrame:
    assert path.exists()

    df: pd.DataFrame = pd.read_csv(path)

    assert not df.empty

    return df


def _build_pipeline() -> Pipeline:
    pipeline: Pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000, random_state=42))
    ])

    assert 'scaler' in pipeline.named_steps
    assert 'clf' in pipeline.named_steps

    return pipeline


def _calculate_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    assert len(y_true) == len(y_prob)
    assert len(y_true) > 0

    auc: float = float(roc_auc_score(y_true, y_prob))
    gini: float = 2.0 * auc - 1.0

    fpr: np.ndarray
    tpr: np.ndarray
    fpr, tpr, _ = roc_curve(y_true, y_prob)

    ks: float = np.max(np.abs(tpr - fpr))

    metrics: dict[str, float] = {
        'auc': auc,
        'gini': gini,
        'ks': ks
    }

    assert 0.0 <= metrics['auc'] <= 1.0
    
    return metrics

def train_and_save(features_path: Path, model_path: Path, metrics_path: Path) -> None:
    assert not model_path.exists()
    assert features_path.exists()

    df: pd.DataFrame = _load_features(features_path).dropna(subset=FEATURE_COLUMNS)

    y: np.ndarray = df[TARGET_COLUMN].values
    X: pd.DataFrame = df[FEATURE_COLUMNS]

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray
    y_test: np.ndarray

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    pipeline: Pipeline = _build_pipeline()
    pipeline.fit(X_train, y_train)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(pipeline, f)
    
    y_prob: np.ndarray = pipeline.predict_proba(X_test)[:, 1]
    metrics: dict[str, float] = _calculate_metrics(y_test, y_prob)

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=4)


def load_model(model_path: Path) -> Pipeline:
    assert model_path.exists()

    with open(model_path, 'rb') as f:
        model: Pipeline = pickle.load(f)
    
    assert isinstance(model, Pipeline)
    
    return model

def predict_proba(customer_id: str, month: int, df_features: pd.DataFrame, model: Pipeline) -> float:
    assert {'customer_id', 'month'}.issubset(set(df_features.columns))

    row:pd.DataFrame = (
        df_features
        .query("customer_id == @customer_id and month == @month")[FEATURE_COLUMNS]
        .fillna(0.0)
    )

    assert not row.empty

    probability: float = float(model.predict_proba(row))[:, 1][0]

    assert 0.0 <= probability <= 1.0

    return probability

def initialize(features_path: Path, model_path: Path, metrics_path: Path) -> Pipeline:
    assert features_path.exists()

    if not model_path.exists():
        train_and_save(features_path, model_path, metrics_path)

    return load_model(model_path)