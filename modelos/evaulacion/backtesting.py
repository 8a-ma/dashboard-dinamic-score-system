import json
import pickle
import sqlite3
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from settings.settings import settings
from modelos.dinamico.kalman import FiltroKalman
from modelos.dinamico.identificacion import load_matrices
from gui.infraestructura.repositorios.decision_repo import insert_decision
from modelos.estatico.logistico import _calculate_metrics, load_model
from gui.infraestructura.repositorios.estado_repo import insert_estimate_state, insert_monthly_state
from modelos.dinamico.controlador import calculate_lqr_gain, decide_credit_limit, dynamic_score, default_cost_matrices


def calculate_month_loss(outstanding_debt: float, default_flag: int, recovery_rate: float = settings.RECOVERY_RATE) -> float:
    assert 0 <= recovery_rate <= 1
    assert outstanding_debt >= 0.0

    return outstanding_debt * (1.0 - recovery_rate) if default_flag == 1 else 0.0


def _normalize_vector(row: pd.Series, scale_params: dict, cols: list[str]) -> np.ndarray:
    assert len(cols) > 0
    assert all(c in scale_params for c in cols)

    return np.array([
        (float(row[c]) - scale_params[c]['mean']) / scale_params[c]['std']
        if scale_params[c]['std'] > 0 else 0.0
        for c in cols
    ], dtype=float)



def simulate_logistic_model(df_features: pd.DataFrame, model: Pipeline) -> pd.DataFrame:
    assert settings.TARGET_COLUMN in df_features.columns
    assert not df_features.empty

    df: pd.DataFrame = df_features.copy()
    df[settings.FEATURE_COLUMNS] = df[settings.FEATURE_COLUMNS].fillna(0.0)

    probs: np.ndarray = model.predict_proba(df[settings.FEATURE_COLUMNS])[:, 1]

    result: pd.DataFrame = df[['customer_id', 'month', 'default_indicator', 'outstanding_debt']].copy()
    result['prob_default'] = probs
    result['loss'] = result.apply(lambda r: calculate_month_loss(float(r['outstanding_debt']), int(r['default_indicator'])), axis=1)

    assert len(result) == len(df_features)
    assert result['prob_default'].between(0.0, 1.0).all()

    return result[['customer_id', 'month', 'prob_default', 'loss', 'default_indicator']]


def _simulate_customer_kalman(df: pd.DataFrame, A: np.ndarray, B: np.ndarray, C: np.ndarray, K: np.ndarray, Q_k: np.ndarray, R_k: np.ndarray, scale_params: dict, credit_limit_max_norm: float) -> list[dict]:
    assert len(df) > 0
    
    kalman: FiltroKalman = FiltroKalman(A, B, C, np.zeros((4, 1)), np.eye(4), Q_k, R_k)
    rows: list[dict] = []

    for _, row in df.sort_values('month').iterrows():
        u_t: np.ndarray = _normalize_vector(row, scale_params, [settings.CONTROL[0]]).reshape(1, 1)
        y_nan: bool = pd.isna(row.get('num_transactions')) or pd.isna(row.get('payment_amount'))
        y_t: np.ndarray = np.full((2, 1), np.nan) if y_nan else _normalize_vector(row, scale_params, settings.OBSERVATIONS).reshape(2, 1)
 
        x_hat, P = kalman.step(u_t, y_t)
        score: float = dynamic_score(x_hat, P, K, credit_limit_max_norm)
        limit: float = decide_credit_limit(K, x_hat, credit_limit_max_norm)
 
        rows.append({
            'customer_id': str(row['customer_id']), 
            'month': int(row['month']),
            'x_hat_debt': float(x_hat[0, 0]),
            'x_hat_income': float(x_hat[1, 0]),
            'x_hat_util': float(x_hat[2, 0]), 
            'x_hat_days': float(x_hat[3, 0]),
            'p_trace': float(np.trace(P)), 
            'score_dinamico': score,
            'limit_recomendado': limit,
            'loss': calculate_month_loss(float(row['outstanding_debt']), int(row['default_indicator'])),
            'default_indicator': int(row['default_indicator'])
        })
 
    return rows

def simulate_dynamic_model(df: pd.DataFrame,A: np.ndarray, B: np.ndarray, C: np.ndarray, K: np.ndarray, Q_k: np.ndarray, R_k: np.ndarray, scale_params: dict) -> pd.DataFrame:
    assert 'customer_id' in df.columns and not df.empty
    assert A.shape == (4, 4) and K.shape == (1, 4)
 
    cl_std: float = scale_params[settings.CONTROL[0]]['std']
    cl_mean: float = scale_params[settings.CONTROL[0]]['mean']
    credit_limit_max_norm: float = (float(df[settings.CONTROL[0]].max()) - cl_mean) / cl_std if cl_std > 0 else 1.0
 
    all_rows: list[dict] = []
    for _, group in df.groupby('customer_id'):
        all_rows.extend(
            _simulate_customer_kalman(group, A, B, C, K, Q_k, R_k, scale_params, credit_limit_max_norm)
        )
 
    result: pd.DataFrame = pd.DataFrame(all_rows)
 
    assert len(result) == len(df)
    assert 'score_dinamico' in result.columns
 
    return result


def calculate_psi(base_scores: np.ndarray, comparison_scores: np.ndarray, bins: int = 10) -> float:
    assert len(base_scores) > 0 and len(comparison_scores) > 0
    assert bins > 0
 
    breakpoints: np.ndarray = np.linspace(0.0, 1.0, bins + 1)
    base_pct: np.ndarray = (np.histogram(base_scores, bins=breakpoints)[0] + 1e-6) / len(base_scores)
    comp_pct: np.ndarray = (np.histogram(comparison_scores, bins=breakpoints)[0] + 1e-6) / len(comparison_scores)
    psi: float = float(np.sum((comp_pct - base_pct) * np.log(comp_pct / base_pct)))
 
    assert np.isfinite(psi)
 
    return psi


def calculate_psi_series(scores_by_month: dict[int, np.ndarray]) -> dict[int, float]:
    assert len(scores_by_month) > 0
 
    months: list[int] = sorted(scores_by_month.keys())
 
    assert len(months) >= 1
 
    base_scores: np.ndarray = scores_by_month[months[0]]
 
    return {month: calculate_psi(base_scores, scores_by_month[month]) for month in months}


def compare_models(logistic_results: pd.DataFrame, dynamic_results: pd.DataFrame) -> dict:
    assert 'prob_default' in logistic_results.columns
    assert 'score_dinamico' in dynamic_results.columns
 
    y_true: np.ndarray = logistic_results[settings.TARGET_COLUMN].values
    log_metrics: dict[str, float] = _calculate_metrics(y_true, logistic_results['prob_default'].values)
    dyn_metrics: dict[str, float] = _calculate_metrics(y_true, 1.0 - dynamic_results['score_dinamico'].values)
 
    log_loss: float = float(logistic_results['loss'].sum())
    dyn_loss: float = float(dynamic_results['loss'].sum())
    reduction: float = (log_loss - dyn_loss) / log_loss if log_loss > 0.0 else 0.0
 
    if reduction < 0.05:
        warnings.warn(f"Dynamic model loss reduction ({reduction:.1%}) < 5% vs baseline", stacklevel=2)
 
    scores_by_month: dict[int, np.ndarray] = {
        int(m): dynamic_results[dynamic_results['month'] == m]['score_dinamico'].values
        for m in sorted(dynamic_results['month'].unique())
    }
    psi_series: dict[int, float] = calculate_psi_series(scores_by_month)
 
    comparison: dict = {
        'logistico': {**log_metrics, 'perdida_total': log_loss},
        'dinamico': {**dyn_metrics, 'perdida_total': dyn_loss},
        'reduccion_perdida': reduction,
        'psi_mensual': {str(k): v for k, v in psi_series.items()}
    }
 
    assert 'reduccion_perdida' in comparison
 
    return comparison


def save_comparison(comparison: dict, path: Path) -> None:
    assert path.parent.exists()
    assert 'logistico' in comparison and 'dinamico' in comparison
 
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(comparison, f, indent=4)
 
    assert path.exists()


def _insert_monthly_states(conn: sqlite3.Connection, df_features: pd.DataFrame) -> None:
    assert 'customer_id' in df_features.columns
    assert 'month' in df_features.columns
 
    available: list[str] = [c for c in settings.MONTHLY_STATE_COLS if c in df_features.columns]
 
    for _, row in df_features[available].iterrows():
        state: dict = {c: (None if pd.isna(row[c]) else row[c]) for c in available}
        if 'transaction_volatility' in state:
            state['transaction_vol'] = state.pop('transaction_volatility')
        insert_monthly_state(conn, state)


def _insert_estimates_and_decisions(conn: sqlite3.Connection, dynamic_results: pd.DataFrame, logistic_results: pd.DataFrame) -> None:
    assert 'x_hat_debt' in dynamic_results.columns
    assert 'prob_default' in logistic_results.columns
 
    merged: pd.DataFrame = dynamic_results.merge(
        logistic_results[['customer_id', 'month', 'prob_default']],
        on=['customer_id', 'month'], how='left'
    )
 
    for _, row in merged.iterrows():
        insert_estimate_state(conn, {
            'customer_id': str(row['customer_id']), 'month': int(row['month']),
            'x_hat_debt': float(row['x_hat_debt']), 'x_hat_income': float(row['x_hat_income']),
            'x_hat_util': float(row['x_hat_util']), 'x_hat_days': float(row['x_hat_days']),
            'p_trace': float(row['p_trace']), 'score_dinamico': float(row['score_dinamico'])
        })
        insert_decision(conn, {
            'customer_id': str(row['customer_id']), 'month': int(row['month']),
            'limit_recomendado': float(row['limit_recomendado']),
            'score_dinamico': float(row['score_dinamico']),
            'score_logistico': float(row.get('prob_default', 0.0))
        })

def populate_sqlite(conn: sqlite3.Connection, dynamic_results: pd.DataFrame, logistic_results: pd.DataFrame, df_features: pd.DataFrame) -> None:
    assert len(dynamic_results) > 0
    assert len(logistic_results) > 0
 
    _insert_monthly_states(conn, df_features)
    _insert_estimates_and_decisions(conn, dynamic_results, logistic_results)


def run_backtesting(features_path: Path, matrices_path: Path, model_path: Path, output_json_path: Path, conn: sqlite3.Connection) -> dict:
    if not model_path.exists():
        raise FileNotFoundError(f"[Fase 3] Modelo no encontrado: {model_path}")
    if not matrices_path.exists():
        raise FileNotFoundError(f"[Fase 3] Matrices no encontradas: {matrices_path}")
    if not features_path.exists():
        raise FileNotFoundError(f"[Fase 3] Features no encontradas: {features_path}")
    
    if not output_json_path.exists():
 
        assert output_json_path.suffix == '.json'
    
        model: Pipeline = load_model(model_path)
    
        A, B, C = load_matrices(matrices_path)
    
        scale_params_path: Path = matrices_path.with_suffix('.json')
        with open(scale_params_path, 'r', encoding='utf-8') as f:
            scale_params: dict = json.load(f)
    
        Q_lqr, R_lqr = default_cost_matrices()
        K: np.ndarray = calculate_lqr_gain(A, B, Q_lqr, R_lqr)
        Q_k: np.ndarray = np.diag([0.01, 0.01, 0.01, 0.01])
        R_k: np.ndarray = np.diag([0.1, 0.1])
    
        df_features: pd.DataFrame = pd.read_csv(features_path)
    
        logistic_results: pd.DataFrame = simulate_logistic_model(df_features, model)
        dynamic_results: pd.DataFrame = simulate_dynamic_model(df_features, A, B, C, K, Q_k, R_k, scale_params)
    
        comparison: dict = compare_models(logistic_results, dynamic_results)
        save_comparison(comparison, output_json_path)
        populate_sqlite(conn, dynamic_results, logistic_results, df_features)
    
        assert output_json_path.exists()
 
    with open(output_json_path, 'r+') as f:
        comparison: dict = f
    
    return comparison

