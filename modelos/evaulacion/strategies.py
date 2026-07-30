import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from sklearn.pipeline import Pipeline
from settings.settings import settings
from modelos.dinamico.kalman import FiltroKalman
from modelos.evaulacion.metrics import calculate_month_loss
from modelos.dinamico.controlador import dynamic_score, decide_credit_limit


class ISimulationStrategy(ABC):
    @abstractmethod
    def simulate(self, df: pd.DataFrame) -> pd.DataFrame:
        pass


class LogisticSimulationStrategy(ISimulationStrategy):
    def __init__(self, model: Pipeline) -> None:
        self._model = model

    def simulate(self, df: pd.DataFrame) -> pd.DataFrame:
        df_copy: pd.DataFrame = df.copy()
        df_copy[settings.FEATURE_COLUMNS] = df_copy[settings.FEATURE_COLUMNS].fillna(0.0)

        probs: np.ndarray = self._model.predict_proba(df_copy[settings.FEATURE_COLUMNS])[:, 1]

        result: pd.DataFrame = df_copy[['customer_id', 'month', 'default_indicator', 'outstanding_debt']].copy()
        result['prob_default'] = probs
        result['loss'] = result.apply(lambda r: calculate_month_loss(float(r['outstanding_debt']), int(r['default_indicator'])), axis=1)

        assert len(result) == len(df_copy)
        assert result['prob_default'].between(0.0, 1.0).all()

        return result[['customer_id', 'month', 'prob_default', 'loss', 'default_indicator']]


class DynamicSimulationStrategy(ISimulationStrategy):
    def __init__(self, A: np.ndarray, B: np.ndarray, C: np.ndarray, K: np.ndarray, Q_k: np.ndarray, R_k: np.ndarray, scale_params: dict) -> None:
        self._A = A
        self._B = B
        self._C = C
        self._K = K
        self._Q_k = Q_k
        self._R_k = R_k

        self._scale_params = scale_params
        self._credit_limit_max_norm = 1.0

    def _simulate_customer_kalman(self, df: pd.DataFrame) -> list[dict]:
        assert not df.empty

        kalman: FiltroKalman = FiltroKalman(
            self._A,
            self._B,
            self._C,
            np.zeros((settings.N_STATES, 1)),
            np.eye(settings.N_STATES),
            self._Q_k,
            self._R_k
        )

        rows: list[dict] = []

        for _, row in df.sort_values('month').iterrows():
            u_t: np.ndarray = self._normalize_vector(row, settings.CONTROL).reshape(-1, 1)
            y_nan: bool = pd.isna(row.get('num_transactions')) or pd.isna(row.get('payment_amount'))
            y_t: np.ndarray = (
                np.full((settings.N_OBSERVATIONS, 1), np.nan) if y_nan
                else self._normalize_vector(row, settings.OBSERVATIONS).reshape(-1, 1)
            )
        
            x_hat, P = kalman.step(u_t, y_t)
            score: float = dynamic_score(x_hat, P, self._K, self._credit_limit_max_norm)
            limit: float = decide_credit_limit(self._K, x_hat, self._credit_limit_max_norm)
            deuda_expuesta: float = min(float(row['outstanding_debt']), limit)
        
            rows.append({
                'customer_id': str(row['customer_id']),
                'month': int(row['month']),
                'x_hat_debt': float(x_hat[0, 0]),
                'x_hat_income': float(x_hat[1, 0]),
                'x_hat_util': float(x_hat[2, 0]),
                'p_trace': float(np.trace(P)),
                'score_dinamico': score,
                'limit_recomendado': limit,
                'loss': calculate_month_loss(deuda_expuesta, int(row['default_indicator'])),
                'default_indicator': int(row['default_indicator'])
            })
         
        return rows

    def _normalize_vector(self, row: pd.Series,  cols: list[str]) -> np.ndarray:
        assert len(cols) > 0
        assert all(c in self._scale_params for c in cols)

        return np.array([
                (float(row[c]) - self._scale_params[c]['mean']) / self._scale_params[c]['std']
                if self._scale_params[c]['std'] > 0 else 0.0
                for c in cols
            ], dtype=float)
        
    def simulate(self, df: pd.DataFrame) -> pd.DataFrame:
        assert 'customer_id' in df.columns and not df.empty
        assert self._A.shape == (settings.N_STATES, settings.N_STATES) and self._K.shape == (1, settings.N_STATES)

        cl_std: float = self._scale_params[settings.CONTROL[0]]['std']
        cl_mean: float = self._scale_params[settings.CONTROL[0]]['mean']

        self._credit_limit_max_norm: float = (float(df[settings.CONTROL[0]].max()) - cl_mean) / cl_std if cl_std > 0 else 1.0

        all_rows: list[dict] = []
        for _, group in df.groupby('customer_id'):
            all_rows.extend(self._simulate_customer_kalman(group))

        result: pd.DataFrame = pd.DataFrame(all_rows)

        assert len(result) == len(df)
        assert 'score_dinamico' in result.columns

        return result
