import json
import warnings
import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from modelos.dinamico.kalman import FiltroKalman
from modelos.dinamico.identificacion import identify, normalize_data
from gui.infraestructura.repositorios.decision_repo import insert_decision
from gui.infraestructura.repositorios.estado_repo import insert_estimate_state
from modelos.dinamico.controlador import default_cost_matrices, calculate_lqr_gain
from modelos.estatico.logistico import predict_proba, _calculate_metrics, load_model


def calcular_perdida_mes(outstanding_debt: float, default: int, tasa_recuperacion: float = 0.30) -> float:
    assert 0 <= tasa_recuperacion <= 1
    assert outstanding_debt >= 0.0

    return outstanding_debt * (1 - tasa_recuperacion) if default == 1 else 0


def simular_modelo_logistico(df_features: pd.DataFrame, model: Pipeline) -> pd.DataFrame:
    assert 'default_indicator' in df_features.columns

    df_indexed = df_features.set_index(['customer_id', 'month'], drop=False)

    prob_defaults:list[float] = []
    perdidas = []

    for c_id, m, debt, default in zip(df_features['customer_id'], df_features['month'], df_features['outstanding_debt'], df_features['default_indicator']):
        prob = predict_proba(c_id, m, df_indexed, model)
        prob_defaults.append(prob)

        perdida = calcular_perdida_mes(debt, default)
        perdidas.append(perdida)
    
    df_result = pd.DataFrame({
        'customer_id': df_features['customer_id'],
        'month': df_features['month'],
        'prob_default': prob_defaults,
        'perdida': perdidas
    })

    return df_result


def simular_modelo_dinamico(df: pd.DataFrame, A: np.ndarray, B: np.ndarray, C: np.ndarray, K: np.ndarray, Q_k: np.ndarray, R_k: np.ndarray) -> pd.DataFrame:
    # - Para cada cliente, instancia `FiltroKalman` y ejecuta Kalman + LQR mes a mes.
    # - Registra en cada paso: `x_hat`, `p_trace`, `score_dinamico`, `limit_recomendado`, `perdida`.
    # - Retorna DataFrame con esas columnas más `customer_id` y `month`.
    ...


def calcular_gini_ks_auc(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    assert len(y_true) == len(y_score)

    return _calculate_metrics(y_true, y_score)


def calcular_psi(score_base: np.ndarray, score_mes: np.ndarray, bins: int = 10) -> float:
    assert len(score_base) * len(score_mes) > 0

    puntos_corte = np.percentile(score_base, np.linspace(0, 100, bins + 1))
    puntos_corte[0] = -np.inf
    puntos_corte[-1] = np.inf

    frec_base, _ = np.histogram(score_base, bins=puntos_corte)
    frec_mes, _ = np.histogram(score_mes, bins=puntos_corte)
    
    prop_base = frec_base / len(score_base)
    prop_mes = frec_mes / len(score_mes)
    
    eps = 1e-4
    prop_base = np.where(prop_base == 0, eps, prop_base)
    prop_mes = np.where(prop_mes == 0, eps, prop_mes)
    
    prop_base /= np.sum(prop_base)
    prop_mes /= np.sum(prop_mes)
    
    psi_vector = (prop_mes - prop_base) * np.log(prop_mes / prop_base)
    psi_total = np.sum(psi_vector)
    
    return float(psi_total)


def calcular_psi_serie(scores_por_mes: dict[int, np.ndarray]) -> dict[int, float]:
    month_base = next(iter(scores_por_mes))
    score_base = scores_por_mes.get(month_base)
    psi_serie: dict = {}
    
    for key, value in scores_por_mes.items():
        if key == month_base: continue

        psi_serie[key] = calcular_psi(score_base, value)
    
    return psi_serie


def comparar_modelos(resultados_logistico: pd.DataFrame, resultados_dinamico: pd.DataFrame, df_features: pd.DataFrame) -> dict:
    # - Computa métricas para ambos modelos (Gini, KS, AUC, pérdida total, PSI mensual).
    # - Si pérdida dinámica no reduce ≥ 5% vs baseline → `warnings.warn(...)` y log (EARS-V05).
    # - Retorna dict para serialización JSON.
    ...


def guardar_comparacion(comparacion: dict, path: Path) -> None:
    assert path.parent.exists()
    
    with open(path, "wb") as f:
        json.dump(comparacion, f, indent=4)


def poblar_sqlite(conn: sqlite3.Connection, resultados_dinamico: pd.DataFrame, decisiones: pd.DataFrame) -> None:
    resultados_dinamico.apply(lambda row: insert_estimate_state(conn, row.to_dict()), axis=1)
    decisiones.apply(lambda row: insert_decision(conn, row.to_dict()), axis=1)


def ejecutar_backtesting(features_path: Path, ruta_matrices: Path, model_pkl_path: Path, ruta_salida_json: Path, conn: sqlite3.Connection) -> None:
    assert features_path.exists()
    assert ruta_matrices.exists()
    assert model_pkl_path.exists()
    assert ruta_salida_json.parent.exists()

    df_features: pd.DataFrame = pd.read_csv(features_path)
    A, B, C = identify(features_path, ruta_matrices)
    model: Pipeline = load_model(model_pkl_path)

    result_logistic_model = simular_modelo_logistico(df_features, model)
    # Obtener K, Q_k, R_k
    result_dinamic_model = simular_modelo_dinamico(df_features, A, B, C, ...)

    results = comparar_modelos(result_logistic_model, result_dinamic_model, df_features)

    # PSI

    guardar_comparacion(results, ruta_salida_json)
