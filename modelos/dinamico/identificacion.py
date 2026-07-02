import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

STATES: list[str] = ['outstanding_debt', 'income', 'utilization_rate', 'days_in_default']
CONTROL: list[str] = ['credit_limit']
OUT_OBS: list[str] = ['num_transactions', 'payment_amount']


def normalizar_datos(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    assert df.shape[1] >= 5
    assert [*STATES, *CONTROL] in df.columns

    df = df[[*STATES, *CONTROL]]

    params: dict[str, dict[str, float]] = {}

    for col in df.columns:
        params[col] = {
            'mean': df[col].mean(),
            'std': df[col].std(ddof=0)
        }
    
    df_zscore: pd.DataFrame = pd.DataFrame(stats.zscore(df, ddof=0), columns=df.columns)

    return df_zscore, params


def construir_matrices_regresion(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    assert ['month', 'customer_id'] in df.columns

    X_in: list[float] | np.ndarray = []
    X_out: list[float] | np.ndarray = []

    for _, group in df.groupby('customer_id'):
        group = group.sort_values('month')

        x_mat: np.ndarray = group[STATES].values
        u_mat: np.ndarray = group[CONTROL].values

        x_t = x_mat[:-1, :]
        u_t = u_mat[:-1, :]

        x_next = x_mat[1:, :]

        xu_t: np.ndarray = np.hstack((x_t, u_t))

        X_in.append(xu_t)
        X_out.append(x_next)
    
    X_in = np.vstack(X_in)
    X_out = np.vstack(X_in)

    assert X_in.shape[0] == X_out.shape[0]

    return X_in, X_out


def identificar_AB(X_in: np.ndarray, X_out: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    assert X_in.shape[1] == 5

    AB = X_out.T @ np.linalg.pinv(X_in.T)

    A = AB[:, :4]
    B = AB[:, 4:5]

    assert A.shape == (4, 4) and B.shape == (4, 1)

    return A, B


def identificar_C(df: pd.DataFrame, A: np.ndarray, B: np.ndarray) -> np.ndarray:
    assert A.shape == (4, 4) and B.shape == (4, 1)

    X = df[STATES].values
    Y = df[OUT_OBS].values

    C = Y.T @ np.linalg.pinv(X.T)

    assert C.shape == (2, 4)

    return C    


def verificar_mse(A: np.ndarray, B: np.ndarray, X_in: np.ndarray, X_out: np.ndarray) -> float:
    X_t = X_in[:, :4]
    U_t = X_in[:, 4:5]

    X_next_pred = X_t @ A.T + U_t @ B.T

    mse = float(np.mean((X_out - X_next_pred) ** 2))

    assert mse < 0.05

    return mse


def guardar_matrices(A: np.ndarray, B: np.ndarray, C: np.ndarray, params_escala: dict, path: Path) -> None:
    np.savez(path, A=A, B=B, C=C)

    json_path = path.with_suffix(".json")

    with open(json_path, "w", encoding='utf-8') as f:
        json.dump(params_escala, f, indent=4)


def cargar_matrices(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert path.exists()

    with np.load(path) as data:
        A: np.ndarray = data["A"]
        B: np.ndarray = data["B"]
        C: np.ndarray = data["C"]
    
    return A, B, C


def identificar(features_path: Path, output_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not output_path.exists():
        df = pd.read_csv(features_path)

        df_zscore, params = normalizar_datos(df)

        X_in, X_out = construir_matrices_regresion(df_zscore)

        A, B = identificar_AB(X_in, X_out)

        C= identificar_C(df_zscore, A, B)

        _ = verificar_mse(A, B, X_in, X_out)

        guardar_matrices(A, B, C, params, output_path)
    
    return cargar_matrices(output_path)