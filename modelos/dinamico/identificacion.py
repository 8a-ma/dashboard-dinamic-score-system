import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from settings.settings import settings


ALL_NUMERIC_COLS: list[str] = [*settings.STATES, *settings.CONTROL, *settings.OBSERVATIONS]


def normalize_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    assert pd.Index(ALL_NUMERIC_COLS).isin(df.columns).all()
    assert not df.empty

    scale_params: dict[str, dict[str, float]]
    df_norm: pd.DataFrame = df.copy()

    means: pd.Series = df[ALL_NUMERIC_COLS].mean()
    stds: pd.Series = df[ALL_NUMERIC_COLS].std(ddof=0)

    df_norm[ALL_NUMERIC_COLS] = (df[ALL_NUMERIC_COLS] - means) / stds

    scale_params = {
        col: {'mean': float(means[col]), 'std': float(stds[col])}
        for col in ALL_NUMERIC_COLS
    }

    return df_norm, scale_params


def build_regression_matrices(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    assert pd.Index(['month', 'customer_id', *settings.STATES, *settings.CONTROL]).isin(df.columns).all()

    df_sorted = df.sort_values(['customer_id', 'month'])

    is_last_month: pd.DataFrame = df_sorted['customer_id'] != df_sorted['customer_id'].shift(-1)

    x_all: np.ndarray = df_sorted[settings.STATES].values
    u_all: np.ndarray = df_sorted[settings.CONTROL].values

    xu_all: np.ndarray = np.hstack((x_all, u_all))

    x_next_all: pd.DataFrame = df_sorted[settings.STATES].shift(-1).values

    valid_rows: np.ndarray = ~is_last_month.values

    X_in: np.ndarray = xu_all[valid_rows]
    X_out: np.ndarray = x_next_all[valid_rows]

    assert X_in.shape[0] == X_out.shape[0]
    assert X_in.shape[1] == 5
    assert not np.isnan(X_out).any()

    return X_in, X_out


def identify_AB(X_in: np.ndarray, X_out: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    assert X_in.shape[1] == 5
    assert X_in.shape[0] == X_out.shape[0]

    AB = X_out.T @ np.linalg.pinv(X_in.T)

    A = AB[:, :4]
    B = AB[:, 4:5]

    eigenvalues: np.ndarray = np.abs(np.linalg.eigvals(A))
    if (eigenvalues > 1.0).any():
        warnings.warn(f"A inestable: eigenvalores {eigenvalues}. Kalman divergira.")

    assert A.shape == (4, 4)
    assert B.shape == (4, 1)

    return A, B


def identify_C(df: pd.DataFrame, A: np.ndarray, B: np.ndarray) -> np.ndarray:
    assert A.shape == (4, 4) and B.shape == (4, 1)
    assert pd.Index([*settings.STATES, *settings.OBSERVATIONS]).isin(df.columns).all()

    X: np.ndarray = df[settings.STATES].values
    Y: np.ndarray = df[settings.OBSERVATIONS].values

    C: np.ndarray = Y.T @ np.linalg.pinv(X.T)

    assert C.shape == (2, 4)

    return C    


def verify_mse(A: np.ndarray, B: np.ndarray, X_in: np.ndarray, X_out: np.ndarray) -> float:
    assert A.shape == (4, 4) and B.shape == (4, 1)
    assert X_in.shape[0] == X_out.shape[0]

    X_t: np.ndarray = X_in[:, :4]
    U_t: np.ndarray = X_in[:, 4:5]

    X_next_pred: np.ndarray = X_t @ A.T + U_t @ B.T

    mse: float = float(np.mean((X_out - X_next_pred) ** 2))

    if mse >= 0.05:
        warnings.warn(f"Reconstruction MSE {mse:.6f} exceeds threshold 0.05")

    return mse


def save_matrices(A: np.ndarray, B: np.ndarray, C: np.ndarray, scale_params: dict[str, dict[str, float]], path: Path) -> None:
    assert A.shape == (4, 4) and B.shape == (4, 1) and C.shape == (2, 4)
    assert path.parent.exists()

    np.savez(path, A=A, B=B, C=C)

    json_path = path.with_suffix(".json")
    with open(json_path, "w", encoding='utf-8') as f:
        json.dump(scale_params, f, indent=4)


def load_matrices(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert path.exists()

    with np.load(path) as data:
        A: np.ndarray = data["A"]
        B: np.ndarray = data["B"]
        C: np.ndarray = data["C"]
    
    assert A.shape == (4, 4) and C.shape == (2, 4)

    return A, B, C


def identify(features_path: Path, output_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert features_path.exists()

    if not output_path.exists():
        df = pd.read_csv(features_path)

        df_norm, scale_params = normalize_data(df)

        X_in, X_out = build_regression_matrices(df_norm)

        A, B = identify_AB(X_in, X_out)
        C = identify_C(df_norm, A, B)

        _ = verify_mse(A, B, X_in, X_out)

        save_matrices(A, B, C, scale_params, output_path)
    
    return load_matrices(output_path)