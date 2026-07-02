"""
| ID | Tarea | Estado | Dependencias | Archivo(s) |
|---|---|---|---|---|
| T2.2.1 | Implementar clase `FiltroKalman.__init__` | ⬜ pendiente | T2.1.7 | `modelos/dinamico/kalman.py` |
| T2.2.2 | Implementar `predecir` (predicción a priori) | ⬜ pendiente | T2.2.1 | `modelos/dinamico/kalman.py` |
| T2.2.3 | Implementar `actualizar` con manejo de NaN (EARS-K03) | ⬜ pendiente | T2.2.2 | `modelos/dinamico/kalman.py` |
| T2.2.4 | Implementar `_simetrizar_covarianza` con detección de autovalores negativos | ⬜ pendiente | T2.2.3 | `modelos/dinamico/kalman.py` |
| T2.2.5 | Implementar `paso` y `ejecutar_secuencia` | ⬜ pendiente | T2.2.4 | `modelos/dinamico/kalman.py` |




#### Parámetros hiperparámetros
- `Q = diag(0.01, 0.01, 0.01, 0.01)` (configurable).
- `R = diag(0.1, 0.1)` (configurable).
- Exponer como argumentos con defaults en `__init__`.
"""

import numpy as np


class FiltroKalman:
    def __init__(self, A: np.ndarray, B: np.ndarray, C: np.ndarray, x0: np.ndarray, P0: np.ndarray, Q: np.ndarray = np.diag([0.01, 0.01, 0.01, 0.01]), R: np.ndarray = np.diag([0.1, 0.1])):
        assert A.shape == (4, 4)
        assert C.shape == (2, 4)
        assert x0.shape == (4, 1)
        assert P0.shape == (4, 4)

        self.A: np.ndarray = A
        self.B: np.ndarray = B
        self.C: np.ndarray = C
        self.Q: np.ndarray = Q
        self.R: np.ndarray = R
        self.x0: np.ndarray = x0
        self.P0: np.ndarray = P0

        self.x_hat: np.ndarray = x0.copy()
        self.P0: np.ndarray = P0.copy()
    
    def predict(self, u_t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x_pred: np.ndarray = self.A @ self.x_hat + self.B @ u_t
        P_pred: np.ndarray = self.A @ self.P @ self.A.T + self.Q

        assert x_pred.shape == (4, 1)

        return x_pred, P_pred
    
    def update(self, y_t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:        
        assert y_t.shape == (2, 1)

        if np.isnan(y_t).any():
            return self.x_hat, self.P

        # x_pred, P_pred = self.predict(np.zeros((self.B.shape[1], 1)))

        S = self.C @ self.P @ self.C.T + self.R
        K = self.P @ self.C.T @ np.linalg.inv(S)

        self.x_hat = self.x_hat + K @ (y_t - self.C @ self.x_hat)

        I = np.eye(self.A.shape[0])
        self.P = (I - K @ self.C) @ self.P

        return self.x_hat, self.P
    
    def _symmetry_covariance(self, P: np.ndarray) -> np.ndarray:
        assert P.shape == (4, 4)

        if np.linalg.eigvals(P).min() < 0:
            P = (P + P.T) / 2.0
        
        return P
    
    def step(self, u_t: np.ndarray, y_t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x_pred, P_pred = self.predict(u_t)
        self.x_hat = x_pred
        self.P = P_pred

        self.x_hat, self.P = self.update()

        self.P = self._symmetry_covariance(self.P)

        return self.x_hat, self.P
    
    def execute_sequence(self, U: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        assert U.shape[0] == Y.shape[0]

        T = U.shape[0]
        X_hat = np.zeros((T, 4))
        P_trace = np.zeros(T)

        for t in range(T):
            # U[t] e Y[t] se extraen y se les da forma de vector columna (N, 1)
            u_t = U[t].reshape(-1, 1)
            y_t = Y[t].reshape(-1, 1)
            
            x_hat, P = self.step(u_t, y_t)
            
            X_hat[t] = x_hat.flatten()
            P_trace[t] = np.trace(P)
            
        return X_hat, P_trace