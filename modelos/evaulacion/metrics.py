import warnings
import numpy as np
import pandas as pd
from settings.settings import settings
from modelos.estatico.logistico import ModelPipelineManager


class MetricsCalculator:
    def calculate_psi(self, base_scores: np.ndarray, comparison_scores: np.ndarray, bins: int = 10) -> float:
        assert len(base_scores) > 0 and len(comparison_scores) > 0
        assert bins > 0

        breakpoints: np.ndarray = np.linspace(0.0, 1.0, bins + 1)
        base_pct: np.ndarray = (np.histogram(base_scores, bins=breakpoints)[0] + 1e-6) / len(base_scores)
        comp_pct: np.ndarray = (np.histogram(comparison_scores, bins=breakpoints)[0] + 1e-6) / len(comparison_scores)
        psi: float = float(np.sum((comp_pct - base_pct) * np.log(comp_pct / base_pct)))

        assert np.isfinite(psi)

        return psi

    def calculate_psi_series(self, scores_by_month: dict[int, np.ndarray]) -> dict[int, float]:
        assert len(scores_by_month) > 0

        months: list[int] = sorted(scores_by_month.keys())
        base_scores: np.ndarray = scores_by_month[months[0]]

        assert len(base_scores) > 0

        return {month: self.calculate_psi(base_scores, scores_by_month[month]) for month in months}

    def compare(self, logistic_df: pd.DataFrame, dynamic_df: pd.DataFrame) -> dict:
        assert 'prob_default' in logistic_df.columns
        assert 'score_dinamico' in dynamic_df.columns

        merged = logistic_df.merge(dynamic_df[['customer_id', 'month', 'score_dinamico', 'loss']], on=['customer_id', 'month'])
        y_true = merged[settings.TARGET_COLUMN].values
        
        log_metrics: dict[str, float] = ModelPipelineManager._calculate_metrics(y_true, logistic_df['prob_default'].values)
        dyn_metrics: dict[str, float] = ModelPipelineManager._calculate_metrics(y_true, 1.0 - dynamic_df['score_dinamico'].values)

        log_loss: float = float(logistic_df['loss'].sum())
        dyn_loss: float = float(dynamic_df['loss'].sum())

        reduction: float = (log_loss - dyn_loss) / log_loss if log_loss > 0.0 else 0.0

        if reduction < 0.05:
            warnings.warn(f"Dynamic model loss reduction ({reduction:.1%}) < 5% vs baseline", stacklevel=2)

        scores_by_month: dict[int, np.ndarray] = {
            int(m): dynamic_df[dynamic_df['month'] == m]['score_dinamico'].values
            for m in sorted(dynamic_df['month'].unique())
        }
        psi_series: dict[int, float] = self.calculate_psi_series(scores_by_month)

        comparison: dict = {
            'logistico': {**log_metrics, 'perdida_total': log_loss},
            'dinamico': {**dyn_metrics, 'perdida_total': dyn_loss},
            'reduccion_perdida': reduction,
            'psi_mensual': {str(k): v for k, v in psi_series.items()}
        }

        assert 'reduccion_perdida' in comparison

        return comparison


def calculate_month_loss(outstanding_debt: float, approved_limit: int, default_flat: float, recovery_rate: float = settings.RECOVERY_RATE) -> float:
    assert 0 <= recovery_rate <= 1
    assert outstanding_debt >= 0.0

    if default_flat == 0:
        return 0.0

    return min(outstanding_debt, approved_limit) * (1.0 - recovery_rate)