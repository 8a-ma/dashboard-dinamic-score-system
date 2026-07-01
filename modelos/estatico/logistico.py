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


def _cargar_features(path: Path) -> pd.DataFrame:
    assert path.exists()

    return pd.read_csv(path)


def _construir_pipeline() -> Pipeline:
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000, random_state=42))
    ])

    assert isinstance(pipeline, Pipeline)
    assert pipeline.named_steps['scaler'] is not None
    assert pipeline.named_steps['clf'] is not None

    return pipeline


def _calcular_metricas(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    assert len(y_true) == len(y_prob)

    auc_score = roc_auc_score(y_true, y_prob)

    gini_coefficient = 2 * auc_score - 1

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    ks_statistic = np.max(np.abs(tpr - fpr))

    metrics: dict = {
        'auc': auc_score,
        'gini': gini_coefficient,
        'ks': ks_statistic
    }

    assert 0 <= metrics['auc'] <= 1
    
    return metrics

def entrenar_y_guardar(features_path: Path, model_path: Path, metrics_path: Path) -> None:
    assert not model_path.exists()

    df = _cargar_features(features_path).copy()

    y: np.ndarray = df['default_indicator'].values
    X: np.ndarray = df[FEATURE_COLUMNS]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

    pipeline = _construir_pipeline()
    pipeline.fit(X_train, y_train)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(pipeline, f)
    
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    metrics = _calcular_metricas(y_test, y_prob)

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=4)


def cargar_modelo(model_path: Path) -> Pipeline:
    assert model_path.exists()

    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    return model

def predict_proba(customer_id: str, month: int, df_features: pd.DataFrame, model: Pipeline) -> float:
    row = df_features.query("customer_id == @customer_id and month == @month")[FEATURE_COLUMNS]

    assert not row.empty

    return model.predict_proba(row)

def inicializar(features_path: Path, model_path: Path, metrics_path: Path) -> Pipeline:
    if not model_path.exists():
        entrenar_y_guardar(features_path, model_path, metrics_path)

    return cargar_modelo(model_path)