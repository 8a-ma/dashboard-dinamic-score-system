import numpy as np
from scipy.linalg import solve_discrete_are
from settings.settings import settings


def calculate_lqr_gain(A: np.ndarray, B: np.ndarray, Q_lqr: np.ndarray, R_lqr: np.ndarray) -> np.ndarray:
    assert A.shape == (settings.N_STATES, settings.N_STATES)
    assert B.shape == (settings.N_STATES, settings.N_CONTROL)

    P: np.ndarray = solve_discrete_are(A, B, Q_lqr, R_lqr)
    K: np.ndarray = np.linalg.inv(R_lqr + B.T @ P @ B) @ (B.T @ P @ A)

    assert K.shape == (settings.N_CONTROL, settings.N_STATES)

    return K


def decide_credit_limit(K: np.ndarray, x_hat: np.ndarray, credit_limit_max: float) -> float:
    assert x_hat.shape == (settings.N_STATES, 1)
    assert K.shape == (settings.N_CONTROL, settings.N_STATES)

    u_b: float = float((-K @ x_hat)[0, 0])
    recommended_limit: float = max(0.0, min(credit_limit_max, u_b))

    assert 0.0 <= recommended_limit <= credit_limit_max

    return recommended_limit


def dynamic_score(x_hat: np.ndarray, P: np.ndarray, K: np.ndarray, credit_limit_max: float) -> float:
    assert 0 < credit_limit_max
    assert x_hat.shape == (settings.N_STATES, 1)

    recommended_limit = decide_credit_limit(K, x_hat, credit_limit_max)
    uncertainty_penalty: float = 0.1
    p_trace: float = float(np.trace(P))
    
    score: float = (recommended_limit / credit_limit_max) * max(0.0, 1.0 - uncertainty_penalty * p_trace)
    score = max(0.0, min(1.0, score))

    assert 0.0 <= score <= 1.0

    return score

def default_cost_matrices() -> tuple[np.ndarray, np.ndarray]:
    Q_lqr: np.ndarray = np.diag([10.0, 1.0, 5.0])
    R_lqr: np.ndarray = np.array([[0.1]])

    assert Q_lqr.shape == (settings.N_STATES, settings.N_STATES)
    assert R_lqr.shape == (settings.N_CONTROL, settings.N_CONTROL)

    return Q_lqr, R_lqr