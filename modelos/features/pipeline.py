import numpy as np
import pandas as pd
from pathlib import Path


def impute_income(df: pd.DataFrame) -> pd.DataFrame:
    assert 'income' in df.columns
    assert 'customer_id' in df.columns and 'month' in df.columns
 
    df = df.sort_values(['customer_id', 'month'])
 
    def fill_income(group: pd.DataFrame) -> pd.DataFrame:
        previous_3_mean: pd.Series = (
            group['income']
            .shift(1)
            .rolling(window=3, min_periods=1)
            .mean()
        )
 
        group['income'] = group['income'].fillna(previous_3_mean)
        group['income'] = group['income'].ffill().bfill()
 
        return group
 
    df = df.groupby('customer_id', group_keys=False).apply(fill_income)
 
    assert df['income'].isna().sum() == 0
 
    return df


def calcular_ratio_deuda_ingreso_ma(df: pd.DataFrame) -> pd.DataFrame:
    assert 'outstanding_debt' in df.columns
    assert 'income' in df.columns
 
    df = df.sort_values(['customer_id', 'month'])
 
    def add_debt_income_ratio_ma(group: pd.DataFrame) -> pd.DataFrame:
        raw_ratio: pd.Series = group['outstanding_debt'] / group['income']
 
        group['ratio_deuda_ingreso_ma'] = (
            raw_ratio.rolling(window=3, min_periods=1).mean()
        )
 
        return group
 
    df = df.groupby('customer_id', group_keys=False).apply(add_debt_income_ratio_ma)
 
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
    assert 'utilization_rate' in df.columns
    assert 'customer_id' in df.columns
 
    df = df.sort_values(['customer_id', 'month'])
 
    def add_utilization_trend(group: pd.DataFrame) -> pd.DataFrame:
        values: np.ndarray = group['utilization_rate'].to_numpy()
        slopes: np.ndarray = np.zeros(len(group), dtype=float)
 
        for i in range(len(group)):
            start: int = max(0, i - 5)
            slopes[i] = _linear_slope(values[start:i + 1])
 
        group['tendencia_utilizacion'] = slopes
 
        return group
 
    df = df.groupby('customer_id', group_keys=False).apply(add_utilization_trend)
 
    assert 'tendencia_utilizacion' in df.columns
    assert len(df) > 0
 
    return df

def calcular_volatilidad_pagos(df: pd.DataFrame) -> pd.DataFrame:
    assert 'payment_amount' in df.columns
    assert 'customer_id' in df.columns
 
    df = df.sort_values(['customer_id', 'month'])
 
    def add_payment_volatility(group: pd.DataFrame) -> pd.DataFrame:
        group['volatilidad_pagos'] = (
            group['payment_amount']
            .shift(1)
            .rolling(window=3, min_periods=1)
            .std()
        )
 
        return group
 
    df = df.groupby('customer_id', group_keys=False).apply(add_payment_volatility)
 
    assert 'volatilidad_pagos' in df.columns
    assert len(df) > 0
 
    return df


def generar_features(input_path: Path, output_path: Path) -> pd.DataFrame:
    assert input_path.exists()
    assert output_path.parent.exists()
 
    raw_transactions: pd.DataFrame = pd.read_csv(input_path)
 
    df: pd.DataFrame = impute_income(raw_transactions)
    df = calcular_ratio_deuda_ingreso_ma(df)
    df = calcular_tendencia_utilizacion(df)
    df = calcular_volatilidad_pagos(df)
 
    df.to_csv(output_path, index=False, encoding='utf-8')
 
    assert output_path.exists()
    assert len(df) == len(raw_transactions)
 
    return df