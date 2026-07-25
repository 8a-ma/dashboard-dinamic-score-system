import sys
import logging
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin


class IncomeImputer(BaseEstimator, TransformerMixin):
    def fit(self, X: pd.DataFrame, y=None) -> 'IncomeImputer':
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        logging.debug(f'{self.__class__.__name__}.{sys._getframe().f_code.co_name} started - rows={len(X)}')

        X = X.copy()

        assert 'income' in X.columns
        assert 'customer_id' in X.columns and 'month' in X.columns

        X = X.sort_values(['customer_id', 'month'])

        previous_3_mean: pd.Series = (
            X.groupby('customer_id', group_keys=False)['income']
            .transform(lambda x: x.shift(1).rolling(window=3, min_periods=1).mean())
        )

        X['income'] = X['income'].fillna(previous_3_mean)
        X['income'] = X.groupby('customer_id', group_keys=False)['income'].transform(lambda x: x.ffill().bfill())

        assert X['income'].isna().sum() == 0

        logging.debug(f'{self.__class__.__name__}.{sys._getframe().f_code.co_name} completed - rows={len(X)}')

        return X


class DebtToIncomeRatioMA(BaseEstimator, TransformerMixin):
    def fit(self, X: pd.DataFrame, y=None) -> 'DebtToIncomeRatioMA':
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        logging.debug(f'{self.__class__.__name__}.{sys._getframe().f_code.co_name} started - rows={len(X)}')

        X = X.copy()
        assert 'customer_id' in X.columns and 'month' in X.columns
        assert 'outstanding_debt' in X.columns and 'income' in X.columns

        X = X.sort_values(['customer_id', 'month'])
        raw_ratio = X['outstanding_debt'] / X['income']

        X['ratio_deuda_ingreso_ma'] = (
            raw_ratio.groupby(X['customer_id'], group_keys=False)
            .transform(lambda x: x.rolling(window=3, min_periods=1).mean())
        )

        assert 'ratio_deuda_ingreso_ma' in X.columns
        assert X['ratio_deuda_ingreso_ma'].ge(0).all()

        logging.debug(f'{self.__class__.__name__}.{sys._getframe().f_code.co_name} completed - rows={len(X)}')

        return X


class UtilizationTrend(BaseEstimator, TransformerMixin):
    def fit(self, X: pd.DataFrame, y=None):
        return self
    
    @staticmethod
    def _linear_slope(window: np.ndarray) -> float:
        assert window.ndim == 1
        assert len(window) >= 0
    
        if len(window) < 2:
            return 0.0
    
        x: np.ndarray = np.arange(len(window))
    
        return float(np.polyfit(x, window, deg=1)[0])


    def transform(self, X:pd.DataFrame) -> pd.DataFrame:
        logging.debug(f'{self.__class__.__name__}.{sys._getframe().f_code.co_name} started - rows={len(X)}')

        X = X.copy()
        assert 'customer_id' in X.columns and 'month' in X.columns
        assert 'utilization_rate' in X.columns

        X = X.sort_values(['customer_id', 'month'])

        def compute_slopes(group_series: pd.Series) -> pd.Series:
            values = group_series.to_numpy()
            n = len(group_series)
            slopes = np.zeros(n, dtype=float)

            for i in range(n):
                start = max(0, i - 5)
                slopes[i] = self._linear_slope(values[start:i + 1])

            return pd.Series(slopes, index=group_series.index)

        X['tendencia_utilizacion'] = (
            X.groupby('customer_id', group_keys=False)['utilization_rate']
            .transform(compute_slopes)
        )

        assert 'tendencia_utilizacion' in X.columns
        assert len(X) > 0

        logging.debug(f'{self.__class__.__name__}.{sys._getframe().f_code.co_name} completed - rows={len(X)}')

        return X


class PaymentVolatility(BaseEstimator, TransformerMixin):
    def fit(self, X: pd.DataFrame, y=None):
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        logging.debug(f'{self.__class__.__name__}.{sys._getframe().f_code.co_name} started - rows={len(X)}')

        X = X.copy()
        
        assert 'customer_id' in X.columns and 'month' in X.columns
        assert 'payment_amount' in X.columns

        X = X.sort_values(['customer_id', 'month'])
 
        X['volatilidad_pagos'] = (
            X.groupby(['customer_id'], group_keys=False)['payment_amount']
            .transform(lambda x: x.shift(1).rolling(window=3, min_periods=1).std())
        )

        assert 'volatilidad_pagos' in X.columns
        assert len(X) > 0

        logging.debug(f'{self.__class__.__name__}.{sys._getframe().f_code.co_name} completed - rows={len(X)}')

        return X
    

def create_pipeline_feature_engineering() -> Pipeline:
    return Pipeline([
        ('imputacion_ingresos', IncomeImputer()),
        ('ratio_deuda_ingreso', DebtToIncomeRatioMA()),
        ('tendencia_utilizacion', UtilizationTrend()),
        ('volatilidad_pagos', PaymentVolatility()),
    ])


def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    assert not df.empty
    assert 'customer_id' in df.columns

    logging.info(f'{sys._getframe().f_code.co_name} started - rows={len(df)} customers={df['customer_id'].nunique()}')

    try:
        pipeline: Pipeline = create_pipeline_feature_engineering()
        df_transformed = pipeline.fit_transform(df)

        assert len(df_transformed) == len(df)

        logging.info(f'{sys._getframe().f_code.co_name} completed - rows={len(df_transformed)} new_cols={['ratio_deuda_ingreso_ma', 'tendencia_utilizacion', 'volatilidad_pagos']}')

        return df_transformed

    except Exception as e:
        logging.error(f'{sys._getframe().f_code.co_name} failed - reason={e}')

        raise