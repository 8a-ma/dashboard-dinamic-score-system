import numpy as np


def denormalize_vector(x_norm: np.ndarray, scale_params: dict, cols: list[str]) -> np.ndarray:
    assert x_norm.shape[0] == len(cols)

    x_real: np.ndarray = np.zeros_like(x_norm)

    for i, col in enumerate(cols):
        mean: float = scale_params[col]['mean']
        std: float = scale_params[col]['std']
        x_real[i] = x_norm[i] * std + mean
        
    return x_real