import warnings
import numpy as np
from settings.settings import settings


class FiltroKalman:
    def __init__(self, A: np.ndarray, B: np.ndarray, C: np.ndarray, x0: np.ndarray, P0: np.ndarray, Q: np.ndarray | None = None, R: np.ndarray | None = None):
        assert A.shape == (settings.N_STATES, settings.N_STATES)
        assert C.shape == (settings.N_OBSERVATIONS, settings.N_STATES)
        assert x0.shape == (settings.N_STATES, settings.N_CONTROL)
        assert P0.shape == (settings.N_STATES, settings.N_STATES)

        self.A: np.ndarray = A
        self.B: np.ndarray = B
        self.C: np.ndarray = C
        self.Q: np.ndarray = Q if Q is not None else np.eye(settings.N_STATES) * 0.01
        self.R: np.ndarray = R if R is not None else np.eye(settings.N_OBSERVATIONS) * 0.1
        self.x_hat: np.ndarray = x0.copy()
        self.P: np.ndarray = P0.copy()
    
    def predict(self, u_t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        assert u_t.shape == (self.B.shape[1], 1)
        assert self.x_hat.shape == (settings.N_STATES, 1)

        x_pred: np.ndarray = self.A @ self.x_hat + self.B @ u_t
        P_pred: np.ndarray = self.A @ self.P @ self.A.T + self.Q

        assert x_pred.shape == (settings.N_STATES, 1)

        return x_pred, P_pred
    
    def update(self, x_pred: np.ndarray, P_pred: np.ndarray, y_t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:        
        assert y_t.shape == (settings.N_OBSERVATIONS, 1)
        assert x_pred.shape == (settings.N_STATES, 1)

        if np.isnan(y_t).any():
            return x_pred, P_pred

        S: np.ndarray = self.C @ P_pred @ self.C.T + self.R
        K: np.ndarray = P_pred @ self.C.T @ np.linalg.pinv(S)

        x_updated: np.ndarray = x_pred + K @ (y_t - self.C @ x_pred)
        I: np.ndarray = np.eye(self.A.shape[0])
        P_updated: np.ndarray = (I - K @ self.C) @ P_pred

        assert x_updated.shape == (settings.N_STATES, 1)

        return x_updated, P_updated
    
    def _symmetrize_covariance(self, P: np.ndarray) -> np.ndarray:
        assert P.shape == (settings.N_STATES, settings.N_STATES)
        assert P.ndim == 2

        if (min_eigenvalue := float(np.linalg.eigvals(P).min().real)) < 0:
            warnings.warn(f"Negative eigenvalue in covariance P ({min_eigenvalue:.6f}). Applying symmetrization.")
            P = (P + P.T) / 2.0
            P = P + (abs(min_eigenvalue) + 1e-6) * np.eye(settings.N_STATES)
        
        return P
    
    def step(self, u_t: np.ndarray, y_t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        assert u_t.shape == (self.B.shape[1], 1)
        assert y_t.shape == (settings.N_OBSERVATIONS, 1)

        x_pred, P_pred = self.predict(u_t)
        x_updated, P_updated = self.update(x_pred, P_pred, y_t)

        self.x_hat = x_updated
        self.P = self._symmetrize_covariance(P_updated)

        return self.x_hat, self.P
    
    def execute_sequence(self, U: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        assert U.shape[0] == Y.shape[0]
        assert U.ndim == 2 and Y.ndim == 2
 
        T: int = U.shape[0]
        X_hat: np.ndarray = np.zeros((T, 4))
        P_trace: np.ndarray = np.zeros(T)
 
        for t in range(T):
            u_t: np.ndarray = U[t].reshape(-1, 1)
            y_t: np.ndarray = Y[t].reshape(-1, 1)
 
            x_hat, P = self.step(u_t, y_t)
 
            X_hat[t] = x_hat.flatten()
            P_trace[t] = float(np.trace(P))
 
        return X_hat, P_trace