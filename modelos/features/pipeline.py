import numpy as np
import pandas as pd
from pathlib import Path


def impute_income(df: pd.DataFrame) -> pd.DataFrame:
    assert 'income' in df.columns
    assert 'customer_id' in df.columns and 'month' in df.columns
 
    df = df.sort_values(['customer_id', 'month'])

    previous_3_mean: pd.Series = (
        df.groupby('customer_id', group_keys=False)['income']
        .transform(lambda x: x.shift(1).rolling(window=3, min_periods=1).mean())
    )

    df['income'] = df['income'].fillna(previous_3_mean)
    df['income'] = df.groupby('customer_id', group_keys=False)['income'].transform(lambda x: x.ffill().bfill())
 
    assert df['income'].isna().sum() == 0
 
    return df


def calcular_ratio_deuda_ingreso_ma(df: pd.DataFrame) -> pd.DataFrame:
    assert 'customer_id' in df.columns and 'month' in df.columns
    assert 'outstanding_debt' in df.columns
    assert 'income' in df.columns
 
    df = df.sort_values(['customer_id', 'month'])
 
    raw_ratio = df['outstanding_debt'] / df['income']
 
    df['ratio_deuda_ingreso_ma'] = (
        raw_ratio.groupby(df['customer_id'], group_keys=False)
        .transform(lambda x: x.rolling(window=3, min_periods=1).mean())
    )
 
    assert 'ratio_deuda_ingreso_ma' in df.columns
    assert df['ratio_deuda_ingreso_ma'].ge(0).all()
 
    return df

def _linear_slope(window: np.ndarray) -> float:
    assert window.ndim == 1
    assert len(window) >= 0
 
    if len(window) < 2:
        return 0.0
 
    x: np.ndarray = np.arange(len(window))
    slope: float = float(np.polyfit(x, window, deg=1)[0])
 
    return slope


def calcular_tendencia_utilizacion(df: pd.DataFrame) -> pd.DataFrame:
    assert 'customer_id' in df.columns and 'month' in df.columns
    assert 'utilization_rate' in df.columns
    assert 'customer_id' in df.columns
 
    df = df.sort_values(['customer_id', 'month'])


    def compute_slopes(group_series: pd.Series) -> pd.Series:
        values: np.ndarray = group_series.to_numpy()
        n: int = len(group_series)
        slopes: np.ndarray = np.zeros(n, dtype=float)
 
        for i in range(n):
            start: int = max(0, i - 5)
            slopes[i] = _linear_slope(values[start:i + 1])
    
        return pd.Series(slopes, index=group_series.index)
 
    df['tendencia_utilizacion'] = (
        df.groupby('customer_id', group_keys=False)['utilization_rate']
        .transform(compute_slopes)
    )
 
    assert 'tendencia_utilizacion' in df.columns
    assert len(df) > 0
 
    return df

def calcular_volatilidad_pagos(df: pd.DataFrame) -> pd.DataFrame:
    assert 'customer_id' in df.columns and 'month' in df.columns
    assert 'payment_amount' in df.columns
    assert 'customer_id' in df.columns
 
    df = df.sort_values(['customer_id', 'month'])
 
    df['volatilidad_pagos'] = (
        df.groupby(['customer_id'], group_keys=False)['payment_amount']
        .transform(lambda x: x.shift(1).rolling(window=3, min_periods=1).std())
    )
 
    assert 'volatilidad_pagos' in df.columns
    assert len(df) > 0
 
    return df

def calcular_days_in_default_lag1(df: pd.DataFrame) -> pd.DataFrame:
    assert 'days_in_default' in df.columns
    assert 'customer_id' in df.columns and 'month' in df.columns
 
    df = df.sort_values(['customer_id', 'month'])
 
    df['days_in_default_lag1'] = (
        df.groupby('customer_id', group_keys=False)['days_in_default']
        .transform(lambda x: x.shift(1).fillna(0.0))
    )
 
    assert 'days_in_default_lag1' in df.columns
    assert df['days_in_default_lag1'].isna().sum() == 0
 
    return df

def generar_features(input_path: Path, output_path: Path) -> pd.DataFrame:
    assert input_path.exists()
    assert output_path.parent.exists()
 
    raw_transactions: pd.DataFrame = pd.read_csv(input_path)
 
    df: pd.DataFrame = impute_income(raw_transactions)
    df = calcular_ratio_deuda_ingreso_ma(df)
    df = calcular_tendencia_utilizacion(df)
    df = calcular_volatilidad_pagos(df)
    # df = calcular_days_in_default_lag1(df)
 
    df.to_csv(output_path, index=False, encoding='utf-8')
 
    assert output_path.exists()
    assert len(df) == len(raw_transactions)
 
    return df