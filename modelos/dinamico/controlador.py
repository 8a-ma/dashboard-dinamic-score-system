import numpy as np
import control as ctrl


def calcular_ganancia_lqr(A: np.ndarray, B: np.ndarray, Q_lqr: np.ndarray, R_lqr: np.ndarray) -> np.ndarray:
    assert A.shape == (4, 4)
    assert B.shape == (4, 1)

    K, _, _ = ctrl.lqr(A, B, Q_lqr, R_lqr)  # Calcula la ganancia (K) y la solución de Riccati (S)

    assert K.shape == (1, 4)

    return K


def decidir_limite(K: np.ndarray, x_hat: np.ndarray, credit_limit_max: float) -> float:
    assert x_hat.shape == (4, 1)
    assert K.shape == (4, 1)

    u_b: float = float(-K @ x_hat)
    recommended_limit: float = max(0.0, min(credit_limit_max, u_b))

    assert 0 <= recommended_limit <= credit_limit_max

    return recommended_limit


def score_dinamico(x_hat: np.ndarray, P: np.ndarray, K: np.ndarray, credit_limit_max: float) -> float:
    assert 0 < credit_limit_max

    recommended_limit = decidir_limite(K, x_hat, credit_limit_max)
    _lambda: float = 0.3

    score = recommended_limit / credit_limit_max * ( 1 - _lambda * ...)

    score: float = max(0.0, min(1.0, score))

    assert 0.0 <= score <= 1.0

    return score

def matrices_costo_default() -> tuple[np.ndarray, np.ndarray]:
    Q_lqr = np.diag(1, 1, 1, 10)
    R_lqr = np.array([0.1])

    return Q_lqr, R_lqr