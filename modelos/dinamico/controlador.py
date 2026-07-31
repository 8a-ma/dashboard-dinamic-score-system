import sys
import logging
import numpy as np
from settings.settings import settings
from scipy.linalg import solve_discrete_are
from utils.hepers import denormalize_vector


def calculate_lqr_gain(A: np.ndarray, B: np.ndarray, Q_lqr: np.ndarray, R_lqr: np.ndarray) -> np.ndarray:
    assert A.shape == (settings.N_STATES, settings.N_STATES)
    assert B.shape == (settings.N_STATES, settings.N_CONTROL)

    try:

        logging.info(f"{sys._getframe().f_code.co_name} started - A.shape={A.shape} B.shape={B.shape}")

        P: np.ndarray = solve_discrete_are(A, B, Q_lqr, R_lqr)
        K: np.ndarray = np.linalg.inv(R_lqr + B.T @ P @ B) @ (B.T @ P @ A)

        assert K.shape == (settings.N_CONTROL, settings.N_STATES)

        logging.info(f"{sys._getframe().f_code.co_name} completed - K.shape={K.shape} norm_K={float(np.linalg.norm(K)):.4f}")

        return K

    except Exception as e:
        logging.error(f"{sys._getframe().f_code.co_name} failed - reason={e}")
        
        raise


def decide_credit_limit(K: np.ndarray, x_hat: np.ndarray, credit_limit_max_norm: float, scale_params: dict) -> float:
    assert x_hat.shape == (settings.N_STATES, 1)
    assert K.shape == (settings.N_CONTROL, settings.N_STATES)
    assert 'credit_limit' in scale_params

    u_b: float = float((-K @ x_hat)[0, 0])
    recommended_limit: float = max(0.0, min(credit_limit_max_norm, u_b))
    recommended_limit_real: float = denormalize_vector(np.array([recommended_limit]), scale_params, ['credit_limit'])[0]

    logging.debug(f"{sys._getframe().f_code.co_name} u_b={u_b:.2f} recommended={recommended_limit_real:.2f} max={credit_limit_max_norm:.2f}")

    assert 0.0 <= recommended_limit_real <= denormalize_vector(np.array([credit_limit_max_norm]), scale_params, ['credit_limit'])[0]

    return recommended_limit_real


def dynamic_score(x_hat: np.ndarray, P: np.ndarray, K: np.ndarray, credit_limit_max_norm: float) -> float:
    assert 0 < credit_limit_max_norm
    assert x_hat.shape == (settings.N_STATES, 1)

    u_b_norm: float = float((-K @ x_hat)[0, 0])
    recommended_norm: float = max(0.0, min(credit_limit_max_norm, u_b_norm))

    uncertainty_penalty: float = 0.1
    p_trace: float = float(np.trace(P))

    score: float = min(1.0, max(0.0, recommended_norm / credit_limit_max_norm))
    score = score * max(0.0, 1.0 - uncertainty_penalty * p_trace)

    logging.debug(f"{sys._getframe().f_code.co_name} score={score:.4f} p_trace={p_trace:.4f} recommended_limit={recommended_norm:.2f}")

    assert 0.0 <= score <= 1.0

    return score

def default_cost_matrices() -> tuple[np.ndarray, np.ndarray]:
    Q_lqr: np.ndarray = np.diag([10.0, 1.0, 5.0])
    R_lqr: np.ndarray = np.array([[0.1]])

    assert Q_lqr.shape == (settings.N_STATES, settings.N_STATES)
    assert R_lqr.shape == (settings.N_CONTROL, settings.N_CONTROL)

    return Q_lqr, R_lqr