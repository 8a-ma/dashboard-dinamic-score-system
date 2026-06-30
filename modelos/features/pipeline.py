"""
| `modelos/features/pipeline.py` | Feature engineering temporal |

| ID | Tarea | Estado | Dependencias | Archivo(s) |
|---|---|---|---|---|
| T1.2.1 | Implementar `imputar_income` | ⬜ pendiente | T1.1.3 | `modelos/features/pipeline.py` |
| T1.2.2 | Implementar `calcular_ratio_deuda_ingreso_ma` | ⬜ pendiente | T1.2.1 | `modelos/features/pipeline.py` |
| T1.2.3 | Implementar `calcular_tendencia_utilizacion` | ⬜ pendiente | T1.2.1 | `modelos/features/pipeline.py` |
| T1.2.4 | Implementar `calcular_volatilidad_pagos` | ⬜ pendiente | T1.2.1 | `modelos/features/pipeline.py` |
| T1.2.5 | Implementar `generar_features` (orquestador) y verificar CSV de salida | ⬜ pendiente | T1.2.2, T1.2.3, T1.2.4 | `modelos/features/pipeline.py`, `db/features_dinamicos.csv` |
"""

import numpy as np
import pandas as pd
from pathlib import Path
from settings.settings import settings


def imputar_ingreso(df: pd.DataFrame) -> pd.DataFrame:
    assert 'income' in df.columns

    df = df.sort_values(['customer_id', 'month'])

    def fill_income(group):
        prev_3_mean = (
            group['income']
            .shift(1)
            .rolling(window=3, min_periods=1)
            .mean()
        )

        group['income'] = group['income'].fillna(prev_3_mean)

        group['income'] = group['income'].ffill().bfill()

        return group

    df = df.groupby('customer_id', group_keys=False).apply(fill_income)

    assert df['income'].isna().sum() == 0

    return df


def calcular_ratio_deuda_ingreso_ma(df: pd.DataFrame) -> pd.DataFrame:
    assert 'outstanding_debt' in df.columns
    assert 'income' in df.columns

    df = df.sort_values(['customer_id', 'month'])

    def add_ratio_deuda_ingreso_ma(group):
        ratio_raw = group["outstanding_debt"] / group["income"]

        group["ratio_deuda_ingreso_ma"] = (
            ratio_raw.rolling(window=3, min_periods=1).mean()
        )

        return group
    
    df = df.groupby('customer_id', group_keys=False).apply(add_ratio_deuda_ingreso_ma)

    assert df['ratio_deuda_ingreso_ma'] >= 0

    return df


def calcular_tendencia_utilizacion(df: pd.DataFrame) -> pd.DataFrame:
    assert 'utilization_rate' in df.columns

    df = df.sort_values(['customer_id', 'month'])

    def add_utilization_rate(group):
        
        y = group['utilization_rate'].to_numpy()
        slopes = np.zeros(len(group), dtype=float)

        for i in range(len(group)):
            start = max(0, i - 5)
            window = y[start:i + 1]

            if len(window) < 2:
                slopes[i] = 0.0

            else:
                # Variable temporal: 0, 1, ..., n-1
                x = np.arange(len(window))
                slopes[i] = np.polyfit(x, window, deg=1)[0]
        
        group['tendencia_utilizacion'] = slopes

        return group
    
    df = df.groupby('customer_id', group_keys=False).apply(add_utilization_rate)

    assert 'tendencia_utilizacion' in df.columns

    return df

def calcular_volatilidad_pagos(df: pd.DataFrame) -> pd.DataFrame:
    assert 'payment_amount' in df.columns

    df = df.sort_values(['customer_id', 'month'])

    def add_volatilidad_pagos(group):
        group['volatilidad_pagos'] = (
            group['payment_amount']
            .shift(1)
            .rolling(window=3, min_periods=1)
            .std()
        )

        return group
    
    df = df.groupby('customer_id', group_keys=False).apply(add_volatilidad_pagos)

    assert 'volatilidad_pagos' in df.columns

    return df


def generar_features(ruta_entrada: Path, ruta_salida: Path) -> pd.DataFrame:
    assert ruta_entrada.exists()
    assert ruta_salida.parent.exists()

    raw_transactions = pd.read_csv(ruta_entrada)

    df = imputar_ingreso(raw_transactions)

    df = calcular_ratio_deuda_ingreso_ma(df)

    df = calcular_tendencia_utilizacion(df)

    df = calcular_volatilidad_pagos(df)

    df.to_csv(ruta_salida, index=False, encoding='utf-8')

    assert ruta_salida.exists()