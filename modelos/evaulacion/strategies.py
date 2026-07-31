import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from sklearn.pipeline import Pipeline
from settings.settings import settings
from utils.hepers import denormalize_vector
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

    def _decide_logistic_limit(self, prob_default: float, original_limit: float, threshold: float = 0.5) -> float:
        assert 0.0 <= prob_default <= 1.0
        assert original_limit >= 0.0

        if prob_default >= threshold:
            return 0.0

        return original_limit * (1.0 - prob_default)
    

    def simulate(self, df: pd.DataFrame) -> pd.DataFrame:
        assert not df.empty
        assert 'month' in df.columns

        df_copy: pd.DataFrame = df.copy()
        df_copy[settings.FEATURE_COLUMNS] = df_copy[settings.FEATURE_COLUMNS].fillna(0.0)

        monthly_results: list = []

        for t in sorted(df_copy['month'].unique()):
            df_t = df_copy[df_copy['month'] == t].copy()

            df_t['prob_default'] = self._model.predict_proba(df_t[settings.FEATURE_COLUMNS])[:, 1]
            df_t['approved_limit'] = [self._decide_logistic_limit(prob, limit) for prob, limit in zip(df_t['prob_default'], df_t['credit_limit'])]
            df_t['loss'] = df_t.apply(
                lambda r: calculate_month_loss(
                    float(r['outstanding_debt']), 
                    r['approved_limit'], 
                    int(r['default_indicator'])
                ), 
                axis=1
            )

            monthly_results.append(df_t)

        result: pd.DataFrame = pd.concat(monthly_results, ignore_index=True)

        assert len(result) == len(df_copy)
        assert result['prob_default'].between(0.0, 1.0).all()

        return result[['customer_id', 'month', 'prob_default', 'approved_limit','loss', 'default_indicator']]


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

        kalman_filters: dict[str, FiltroKalman] = {}
        all_results: list[dict] = []

        customer_groups = {cid: group.set_index('month') for cid, group in df.groupby('customer_id')}

        for cid in sorted(df['customer_id'].unique()):
            df_client = customer_groups[cid]

            kalman = FiltroKalman(
                            self._A,
                            self._B,
                            self._C,
                            np.zeros((settings.N_STATES, 1)),
                            np.eye(settings.N_STATES),
                            self._Q_k,
                            self._R_k
                        )

            for t in sorted(df['month'].unique()):
                if t not in df_client.index:
                    continue

                row = df_client.loc[t]

                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]

                u_t: np.ndarray = self._normalize_vector(row, settings.CONTROL).reshape(-1, 1)
                y_nan: bool = pd.isna(row.get('num_transactions')) or pd.isna(row.get('payment_amount'))
                y_t: np.ndarray = np.full((settings.N_OBSERVATIONS, 1), np.nan) if y_nan else self._normalize_vector(row, settings.OBSERVATIONS).reshape(-1, 1)

                x_hat, P = kalman.step(u_t, y_t)
                approved_limit: float = decide_credit_limit(self._K, x_hat, self._credit_limit_max_norm, self._scale_params)
                score: float = dynamic_score(x_hat, P, self._K, self._credit_limit_max_norm)
                loss: float = calculate_month_loss(
                    row['outstanding_debt'],
                    approved_limit,
                    row['default_indicator']
                )

                all_results.append({
                    'customer_id': str(cid),
                    'month': int(t),
                    'x_hat_debt': float(x_hat[0, 0]),
                    'x_hat_income': float(x_hat[1, 0]),
                    'x_hat_util': float(x_hat[2, 0]),
                    'p_trace': float(np.trace(P)),
                    'score_dinamico': score,
                    'approved_limit': approved_limit,
                    'loss': loss,
                    'default_indicator': int(row['default_indicator'])
                })

        result: pd.DataFrame = pd.DataFrame(all_results)

        assert len(result) == len(df)
        assert 'score_dinamico' in result.columns

        return result