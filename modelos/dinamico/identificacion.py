import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from settings.settings import settings
from utils.file_helpers import save_npz, load_npz, save_dict_to_json, load_json_to_dict, load_csv_to_df


ALL_NUMERIC_COLS: list[str] = [*settings.STATES, *settings.CONTROL, *settings.OBSERVATIONS]


class StateSpaceModel:
    def __init__(self) -> None:
        self.A: np.ndarray | None = None
        self.B: np.ndarray | None = None
        self.C: np.ndarray | None = None
        self.scale_params: dict[str, dict[str, float]] | None = None


    def _normalize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        assert pd.Index(ALL_NUMERIC_COLS).isin(df.columns).all()
        assert not df.empty

        df_norm: pd.DataFrame = df.copy()

        means: pd.Series = df[ALL_NUMERIC_COLS].mean()
        stds: pd.Series = df[ALL_NUMERIC_COLS].std(ddof=0)
        safe_stds: pd.Series = stds.replace(0.0, 1.0)

        df_norm[ALL_NUMERIC_COLS] = (df[ALL_NUMERIC_COLS] - means) / safe_stds

        self.scale_params = {
            col: {'mean': float(means[col]), 'std': float(stds[col])}
            for col in ALL_NUMERIC_COLS
        }

        assert not df_norm[ALL_NUMERIC_COLS].isnull().any().any()

        return df_norm


    def _build_regression_matrices(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        assert pd.Index(['month', 'customer_id', *settings.STATES, *settings.CONTROL]).isin(df.columns).all()
        assert not df.empty

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
        assert X_in.shape[1] == settings.N_STATES + settings.N_CONTROL
        assert not np.isnan(X_out).any()

        return X_in, X_out


    def _identify_AB(self, X_in: np.ndarray, X_out: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        assert X_in.shape[1] == settings.N_STATES + settings.N_CONTROL
        assert X_in.shape[0] == X_out.shape[0]

        AB = X_out.T @ np.linalg.pinv(X_in.T)

        A: np.ndarray = AB[:, :settings.N_STATES]
        B: np.ndarray = AB[:, settings.N_STATES:settings.N_STATES + settings.N_CONTROL]

        print(A.shape)

        eigenvalues: np.ndarray = np.abs(np.linalg.eigvals(A))
        if (eigenvalues > 1.0).any():
            warnings.warn(f"A inestable: eigenvalores {eigenvalues}. Kalman divergira.")

        assert A.shape == (settings.N_STATES, settings.N_STATES)
        assert B.shape == (settings.N_STATES, settings.N_CONTROL)

        return A, B


    def _identify_C(self, df: pd.DataFrame) -> np.ndarray:
        assert pd.Index([*settings.STATES, *settings.OBSERVATIONS]).isin(df.columns).all()

        X: np.ndarray = df[settings.STATES].values
        Y: np.ndarray = df[settings.OBSERVATIONS].values

        C: np.ndarray = Y.T @ np.linalg.pinv(X.T)

        assert C.shape == (settings.N_OBSERVATIONS, settings.N_STATES)

        return C


    def _verify_mse(self, X_in: np.ndarray, X_out: np.ndarray) -> float:
        assert self.A.shape == (settings.N_STATES, settings.N_STATES)
        assert self.B.shape == (settings.N_STATES, settings.N_CONTROL)
        assert X_in.shape[0] == X_out.shape[0]

        X_t: np.ndarray = X_in[:, :settings.N_STATES]
        U_t: np.ndarray = X_in[:, settings.N_STATES:settings.N_STATES + settings.N_CONTROL]

        X_next_pred: np.ndarray = X_t @ self.A.T + U_t @ self.B.T

        mse: float = float(np.mean((X_out - X_next_pred) ** 2))

        if mse >= 0.05:
            warnings.warn(f"Reconstruction MSE {mse:.6f} exceeds threshold 0.05")

        return mse

    def fit(self, df: pd.DataFrame) -> 'StateSpaceModel':
        df_norm = self._normalize_data(df)
        X_in, X_out = self._build_regression_matrices(df_norm)

        self.A, self.B = self._identify_AB(X_in, X_out)
        self.C = self._identify_C(df_norm)

        _ = self._verify_mse(X_in, X_out)

        return self

    def save(self) -> None:
        assert self.A is not None and self.B is not None and self.C is not None
        assert self.scale_params is not None

        save_npz(settings.MATRIX_SYSTEM_PATH, A=self.A, B=self.B, C=self.C)
        save_dict_to_json(self.scale_params, settings.MATRIX_SYSTEM_SCALE_PATH)

    @classmethod
    def load(cls) -> 'StateSpaceModel':
        matrix: dict[str, np.ndarray] = load_npz(settings.MATRIX_SYSTEM_PATH)

        model = cls()
        model.A = matrix['A']
        model.B = matrix['B']
        model.C = matrix['C']

        model.scale_params = load_json_to_dict(settings.MATRIX_SYSTEM_SCALE_PATH)

        assert model.A.shape == (settings.N_STATES, settings.N_STATES)
        assert model.B.shape == (settings.N_STATES, settings.N_CONTROL)
        assert model.C.shape == (settings.N_OBSERVATIONS, settings.N_STATES)

        return model


def identify(force_train: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert settings.FEATURES_PATH.exists()

    model_exists = settings.MATRIX_SYSTEM_PATH.exists() and settings.MATRIX_SYSTEM_SCALE_PATH.exists()

    if force_train or not model_exists:
        df = load_csv_to_df(settings.FEATURES_PATH)

        model = StateSpaceModel()
        model.fit(df)
        model.save()

    model = StateSpaceModel.load()
    assert model.A is not None and model.B is not None and model.C is not None

    return model.A, model.B, model.C