# Plan de Implementación — dashboard-dinamic-score-system

---

## Índice

1. [Visión general del sistema](#1-visión-general-del-sistema)
2. [Restricciones y reglas de codificación](#2-restricciones-y-reglas-de-codificación)
3. [Estructura de archivos objetivo](#3-estructura-de-archivos-objetivo)
4. [Orden de implementación global](#4-orden-de-implementación-global)
5. [Fase 0 — Bootstrapping del proyecto](#fase-0--bootstrapping-del-proyecto)
6. [Fase 1 — Dataset y features dinámicos](#fase-1--dataset-y-features-dinámicos)
7. [Fase 2 — Modelo dinámico (Kalman + LQR)](#fase-2--modelo-dinámico-kalman--lqr)
8. [Fase 3 — Backtesting y comparación](#fase-3--backtesting-y-comparación)
9. [Fase 4 — Interfaz gráfica NiceGUI](#fase-4--interfaz-gráfica-nicegui)
10. [Infraestructura SQLite y repositorios](#infraestructura-sqlite-y-repositorios)
11. [Tests pytest](#tests-pytest)
12. [Mapa de dependencias entre artefactos](#mapa-de-dependencias-entre-artefactos)
13. [Criterios de aceptación globales](#criterios-de-aceptación-globales)

---

## 1. Visión general del sistema

El sistema es una **aplicación monolítica Python 3.12** que construye un motor de scoring crediticio dinámico. En lugar de producir probabilidades estáticas, modela al cliente como un sistema dinámico con retroalimentación (banco → cliente → banco), estimando estado con Kalman y optimizando decisiones con LQR.

### Componentes macro

| Componente | Responsabilidad |
|---|---|
| `scripts/transacciones.py` | Generación de datos sintéticos (500 clientes × 24 meses) |
| `modelos/features/pipeline.py` | Feature engineering temporal |
| `modelos/estatico/logistico.py` | Baseline logístico, serialización en `.pkl` |
| `modelos/dinamico/identificacion.py` | Estimación de matrices A, B, C del sistema |
| `modelos/dinamico/kalman.py` | Filtro de Kalman discreto |
| `modelos/dinamico/controlador.py` | LQR, score dinámico normalizado |
| `modelos/evaluacion/backtesting.py` | Simulación mes a mes, comparación de modelos |
| `gui/infraestructura/db.py` | Inicialización SQLite, tablas |
| `gui/infraestructura/repositorios/` | Repositorios DDD (Customer, Decision, Estado) |
| `gui/cliente/` | Dashboard del cliente (NiceGUI) |
| `gui/banco/` | Dashboard del banco (NiceGUI) |
| `tests/` | Suite pytest |
| `settings/settings.py` | Clase con configuraciones globales para constantes |

---

## 2. Restricciones y reglas de codificación

Estas reglas aplican a **cada función de cada archivo** y deben verificarse antes de marcar una tarea como completada:

| Regla | Descripción |
|---|---|
| **Longitud** | Ninguna función supera 60 líneas |
| **Tipado explícito** | Toda variable y parámetro lleva anotación de tipo |
| **Aserciones** | Mínimo 2 aserciones por función (precondiciones/invariantes) |
| **Legibilidad** | Código autoexplicativo; mínimos comentarios inline |
| **Unidades CSS** | Solo `rem`, base `font-size: 10px` |
| **CSS Modules** | Clases prefijadas por componente; sin colisiones |
| **Seed fijo** | `SEED = 42` antes de cualquier `numpy.random` |
| **Singleton .pkl** | Modelo logístico entrenado una sola vez; reutilizado en adelante |
| **FileNotFoundError** | Si falta cualquier dependencia de fase, lanzar con ruta y fase |

---

## 3. Estructura de archivos objetivo

```
dashboard-dinamic-score-system/
├── AGENTS.md
├── db/
│   ├── credito.db                  # SQLite principal
│   ├── raw_transactions.csv        # Salida Fase 1.1
│   ├── features_dinamicos.csv      # Salida Fase 1.2
│   ├── matrices_sistema.npz        # Salida Fase 2.1
│   ├── modelo_logistico.pkl        # Salida Fase 1.3
│   ├── metricas_baseline.json      # Salida Fase 1.3
│   └── comparacion_modelos.json    # Salida Fase 3
├── modelos/
│   ├── __init__.py
│   ├── features/
│   │   ├── __init__.py
│   │   └── pipeline.py
│   ├── estatico/
│   │   ├── __init__.py
│   │   └── logistico.py
│   ├── dinamico/
│   │   ├── __init__.py
│   │   ├── identificacion.py
│   │   ├── kalman.py
│   │   └── controlador.py
│   └── evaluacion/
│       ├── __init__.py
│       └── backtesting.py
├── gui/
│   ├── global.css
│   ├── infraestructura/
│   │   ├── __init__.py
│   │   ├── db.py
│   │   └── repositorios/
│   │       ├── __init__.py
│   │       ├── cliente_repo.py
│   │       ├── decision_repo.py
│   │       └── estado_repo.py
│   ├── cliente/
│   │   ├── styles.css
│   │   ├── componentes/
│   │   │   ├── __init__.py
│   │   │   ├── score_gauge.py
│   │   │   ├── trayectoria_chart.py
│   │   │   ├── tabla_transacciones.py
│   │   │   └── selector_cliente.py
│   │   └── paginas/
│   │       ├── __init__.py
│   │       └── dashboard_cliente.py
│   └── banco/
│       ├── styles.css
│       ├── componentes/
│       │   ├── __init__.py
│       │   ├── panel_metricas.py
│       │   ├── autovalores_plot.py
│       │   ├── psi_chart.py
│       │   ├── features_agregados.py
│       │   ├── simulacion_contrafactual.py
│       │   └── filtro_arquetipo.py
│       └── paginas/
│           ├── __init__.py
│           └── dashboard_banco.py
├── scripts/
│   └── transacciones.py
├── settings/
│   └── settings.py
├── tests/
│   ├── conftest.py
│   ├── test_features.py
│   ├── test_kalman.py
│   ├── test_controlador.py
│   ├── test_logistico.py
│   └── test_integracion.py
└── docs/
    ├── plan-implementacion.md      ← este archivo
    └── tareas.md
```

---

## 4. Orden de implementación global

La secuencia respeta el grafo de dependencias entre artefactos de salida:

```
Fase 0 (setup)
    ↓
Fase 1.1 (transacciones.py → raw_transactions.csv)
    ↓
Fase 1.2 (pipeline.py → features_dinamicos.csv)
    ↓
Fase 1.3 (logistico.py → modelo_logistico.pkl + metricas_baseline.json)
    ↓
Fase 2.1 (identificacion.py → matrices_sistema.npz)
    ↓
Fase 2.2 (kalman.py)
    ↓
Fase 2.3 (controlador.py)
    ↓
Infraestructura SQLite (db.py + repositorios)
    ↓
Fase 3 (backtesting.py → comparacion_modelos.json + poblar credito.db)
    ↓
Fase 4 (GUI NiceGUI — cliente y banco)
    ↓
Tests (conftest → unit tests → test integración)
```

---

## Fase 0 — Bootstrapping del proyecto

### Objetivo
Dejar la estructura de carpetas, dependencias y configuración base lista para que las fases siguientes puedan arrancar.

### Tareas

#### T0.1 — Estructura de directorios
- Crear todos los directorios del árbol de archivos objetivo.
- Crear archivos `__init__.py` vacíos en todos los paquetes Python.
- Crear archivos CSS vacíos en sus rutas finales (`global.css`, `cliente/styles.css`, `banco/styles.css`).

#### T0.2 — Archivo de dependencias
- Crear `requirements.txt` con versiones fijadas:
  ```
  numpy>=1.26
  pandas>=2.2
  scikit-learn>=1.4
  python-control>=0.9
  nicegui>=1.4
  scipy>=1.12
  ```
- Verificar que `python-control` incluye `dare` (ecuación de Riccati discreta).

#### T0.3 — Documentación inicial
- Crear `docs/tareas.md` con tabla de tareas y columna de estado (`pendiente / en curso / completado`).
- Confirmar que `docs/plan-implementacion.md` (este archivo) está en su ruta.

#### T0.4 — Validación mínima
- Ejecutar `python --version` → confirmar 3.12.
- Ejecutar `pip install -r requirements.txt`.
- Confirmar que `import nicegui`, `import control`, `import sklearn` no lanzan error.

---

## Fase 1 — Dataset y features dinámicos

### 1.1 — Generación de datos sintéticos

**Archivo:** `scripts/transacciones.py`  
**Salida:** `db/raw_transactions.csv`  
**EARS:** D01–D05

#### Descripción funcional
El script ya existe (proporcionado). Genera 500 clientes × 24 meses con 5 arquetipos (`good`, `recurrent`, `over`, `fraud`, `low`) según las proporciones definidas. Cada fila es un registro mensual con: `customer_id`, `month`, `income`, `credit_limit`, `utilization_rate`, `outstanding_debt`, `payment_amount`, `days_in_default`, `num_transactions`, `transaction_volatility`, `default_indicator`.

#### Tareas de implementación

**T1.1.1 — Revisar y consolidar el script existente**
- Verificar que el script ya cumpla D01, D02, D05 (500 clientes, 5 arquetipos, `SEED=42`).
- Añadir flag `--force` en `argparse`: si `db/raw_transactions.csv` ya existe y no se pasa `--force`, el script imprime mensaje y termina sin regenerar (D04).
- Confirmar encoding UTF-8 en `to_csv` (ya presente).

**T1.1.2 — Añadir aserciones al script**
- Al inicio de la función generadora: `assert N_CUSTOMERS == 500`.
- Tras generar `df`: `assert len(df) == 500 * 24` y `assert set(df['customer_id'].str[0]) == {'C'}`.

**T1.1.3 — Verificación de ejecución**
- Ejecutar el script y confirmar que `db/raw_transactions.csv` aparece con 12 000 filas (500 × 24).
- Verificar distribución de arquetipos en el CSV: `good≈200`, `recurrent≈125`, `over≈100`, `fraud≈25`, `low≈50`.

---

### 1.2 — Feature Engineering Dinámico

**Archivo:** `modelos/features/pipeline.py`  
**Salida:** `db/features_dinamicos.csv`  
**EARS:** F01–F06

#### Descripción funcional
Toma `raw_transactions.csv`, computa tres features temporales sobre ventanas móviles y exporta el dataset enriquecido. `customer_id` y `month` nunca se modifican (índice compuesto).

#### Funciones a implementar

**`imputar_income(df: pd.DataFrame) -> pd.DataFrame`**
- Precondición: `assert 'income' in df.columns`.
- Por cliente, rellenar NaN de `income` con la media de los 3 meses previos del mismo cliente.
- Si no hay meses previos disponibles, usar `ffill` y luego `bfill` como fallback.
- Postcondición: `assert df['income'].isna().sum() == 0` (o documentar casos irreducibles).

**`calcular_ratio_deuda_ingreso_ma(df: pd.DataFrame) -> pd.DataFrame`**
- Precondición: `assert 'outstanding_debt' in df.columns and 'income' in df.columns`.
- Por cliente ordenado por `month`, computar `ratio_raw = outstanding_debt / income`.
- Media móvil de 3 meses (`min_periods=1`).
- Asignar a columna `ratio_deuda_ingreso_ma`.
- Postcondición: resultado ≥ 0 en toda la columna (después de imputación de income).

**`calcular_tendencia_utilizacion(df: pd.DataFrame) -> pd.DataFrame`**
- Precondición: `assert 'utilization_rate' in df.columns`.
- Por cliente, para cada `month`, calcular pendiente de regresión lineal de `utilization_rate` sobre los últimos 6 meses disponibles (ventana deslizante).
- Si hay menos de 2 puntos, pendiente = 0.0.
- Asignar a columna `tendencia_utilizacion`.

**`calcular_volatilidad_pagos(df: pd.DataFrame) -> pd.DataFrame`**
- Precondición: `assert 'payment_amount' in df.columns`.
- Por cliente, desviación estándar móvil de 3 meses de `payment_amount` (`min_periods=1`, `std`).
- Asignar a columna `volatilidad_pagos`.

**`generar_features(ruta_entrada: Path, ruta_salida: Path) -> pd.DataFrame`**
- Orquestador: carga CSV, llama las 4 funciones anteriores en orden, exporta a `ruta_salida` con `index=False`, encoding UTF-8.
- Precondición: `assert ruta_entrada.exists()`.
- Postcondición: `assert ruta_salida.exists()`.

#### Consideraciones de implementación
- Todas las operaciones de ventana deben hacerse **dentro de cada `customer_id`** usando `groupby(...).transform(...)` o `groupby(...).apply(...)`.
- La pendiente de regresión lineal se puede calcular con `np.polyfit(x, y, 1)[0]` dentro de una función auxiliar `_pendiente_lineal(series: pd.Series) -> float`.

---

### 1.3 — Modelo Logístico Baseline

**Archivo:** `modelos/estatico/logistico.py`  
**Salidas:** `db/modelo_logistico.pkl`, `db/metricas_baseline.json`  
**EARS:** B01–B06

#### Descripción funcional
Modelo logístico scikit-learn entrenado una sola vez, serializado en `.pkl`. Expone `predict_proba(customer_id, month) -> float`. Métricas (Gini, KS, AUC) calculadas al entrenar y guardadas en JSON.

#### Funciones a implementar

**`_cargar_features(ruta: Path) -> pd.DataFrame`**
- Carga `features_dinamicos.csv`.
- Precondición: `assert ruta.exists()`.
- Retorna DataFrame con columnas necesarias.

**`_construir_pipeline() -> Pipeline`**
- Retorna `Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(max_iter=1000))])`.
- Las features de entrada son: `ratio_deuda_ingreso_ma`, `tendencia_utilizacion`, `volatilidad_pagos`, `utilization_rate`, `days_in_default`, `num_transactions`.

**`_calcular_metricas(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]`**
- Calcula AUC-ROC, Gini (= 2·AUC − 1), KS (max diferencia entre TPR y FPR).
- Precondición: `assert len(y_true) == len(y_prob)`.
- Postcondición: `assert 0 <= metricas['auc'] <= 1`.
- Retorna dict con claves `auc`, `gini`, `ks`.

**`entrenar_y_guardar(ruta_features: Path, ruta_modelo: Path, ruta_metricas: Path) -> None`**
- Split estratificado 80/20 por `default_indicator`.
- Entrena pipeline, serializa con `pickle` en `ruta_modelo`.
- Calcula métricas sobre test set y guarda JSON en `ruta_metricas`.
- Precondición: `assert not ruta_modelo.exists()` (no re-entrenar si ya existe).

**`cargar_modelo(ruta_modelo: Path) -> Pipeline`**
- Deserializa y retorna el pipeline.
- Precondición: `assert ruta_modelo.exists()`.

**`predict_proba(customer_id: str, month: int, df_features: pd.DataFrame, modelo: Pipeline) -> float`**
- Filtra la fila `(customer_id, month)` del DataFrame de features.
- Precondición: `assert not fila.empty`.
- Retorna `float` en `[0, 1]` representando P(default).

**`inicializar(ruta_features: Path, ruta_modelo: Path, ruta_metricas: Path) -> Pipeline`**
- Si `ruta_modelo` existe → carga y retorna.
- Si no → llama `entrenar_y_guardar` y retorna modelo cargado.
- Implementa EARS-G07 y EARS-B04.

---

## Fase 2 — Modelo dinámico (Kalman + LQR)

### Prerequisito de Fase 2
Verificar existencia de `db/raw_transactions.csv` y `db/features_dinamicos.csv`. Si no existen, lanzar `FileNotFoundError` con ruta y fase responsable (EARS-DEP01).

---

### 2.1 — Identificación del sistema

**Archivo:** `modelos/dinamico/identificacion.py`  
**Salida:** `db/matrices_sistema.npz`  
**EARS:** S01–S05

#### Descripción funcional
Ajusta por regresión el modelo de espacio de estado discreto:

```
x_{t+1} = A·x_t + B·u_t
y_t      = C·x_t
```

Donde:
- **Estado** `x_c ∈ ℝ⁴`: `[outstanding_debt, income, utilization_rate, days_in_default]`
- **Control** `u_t ∈ ℝ¹`: `credit_limit`
- **Salidas observables** `y_t ∈ ℝ²`: `[num_transactions, payment_amount]`

#### Funciones a implementar

**`normalizar_datos(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]`**
- Normaliza las 4 variables de estado y el control a escala `[0, 1]` o media 0 / std 1.
- Retorna DataFrame normalizado y diccionario de parámetros de escala para desnormalizar.
- Precondición: `assert df.shape[1] >= 5`.

**`construir_matrices_regresion(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]`**
- Por cada cliente, para meses `t = 1..23`, construye pares `(x_t, u_t)` → `x_{t+1}`.
- Empila en matrices `X_in ∈ ℝ^{N×5}` (estado + control) y `X_out ∈ ℝ^{N×4}` (siguiente estado).
- Precondición: `assert X_in.shape[0] == X_out.shape[0]`.

**`identificar_AB(X_in: np.ndarray, X_out: np.ndarray) -> tuple[np.ndarray, np.ndarray]`**
- Resuelve `[A | B] = X_out.T @ np.linalg.pinv(X_in.T)` (regresión mínimos cuadrados).
- `A ∈ ℝ^{4×4}`, `B ∈ ℝ^{4×1}`.
- Precondición: `assert X_in.shape[1] == 5`.
- Postcondición: `assert A.shape == (4, 4) and B.shape == (4, 1)`.

**`identificar_C(df: pd.DataFrame, A: np.ndarray, B: np.ndarray) -> np.ndarray`**
- Ajusta `C ∈ ℝ^{2×4}` tal que `y_t ≈ C·x_t` por mínimos cuadrados.
- Precondición: `assert A.shape == (4, 4)`.
- Postcondición: `assert C.shape == (2, 4)`.

**`verificar_mse(A: np.ndarray, B: np.ndarray, X_in: np.ndarray, X_out: np.ndarray) -> float`**
- Calcula MSE de reconstrucción en escala normalizada.
- Precondición: matrices consistentes en dimensiones.
- Postcondición: `assert mse < 0.05` (EARS-S04) — si falla, loguea advertencia.

**`guardar_matrices(A, B, C, params_escala, ruta: Path) -> None`**
- Serializa con `np.savez(ruta, A=A, B=B, C=C)` y guarda `params_escala` en JSON auxiliar.

**`cargar_matrices(ruta: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]`**
- Carga y retorna `(A, B, C)`.
- Precondición: `assert ruta.exists()`.

**`identificar(ruta_features: Path, ruta_salida: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]`**
- Orquestador: si `ruta_salida` existe → carga. Si no → ejecuta pipeline completo y guarda.

---

### 2.2 — Filtro de Kalman

**Archivo:** `modelos/dinamico/kalman.py`  
**EARS:** K01–K05

#### Descripción funcional
Filtro de Kalman discreto que estima el estado del cliente `x̂_t` a partir de observaciones parciales `y_t = [num_transactions, payment_amount]`. Implementado como clase para mantener estado entre llamadas.

#### Clase `FiltroKalman`

**`__init__(self, A, B, C, Q, R, x0, P0)`**
- `A ∈ ℝ^{4×4}`, `B ∈ ℝ^{4×1}`, `C ∈ ℝ^{2×4}`.
- `Q ∈ ℝ^{4×4}` diagonal (ruido de proceso), `R ∈ ℝ^{2×2}` diagonal (ruido de medición).
- `x0 ∈ ℝ^{4×1}` estado inicial, `P0 ∈ ℝ^{4×4}` covarianza inicial.
- Precondición: `assert A.shape == (4, 4)` y `assert C.shape == (2, 4)`.

**`predecir(self, u_t: np.ndarray) -> tuple[np.ndarray, np.ndarray]`**
- Predicción: `x̂_{t|t-1} = A·x̂_{t-1} + B·u_t`.
- `P_{t|t-1} = A·P_{t-1}·A^T + Q`.
- Retorna `(x_pred, P_pred)`.
- Postcondición: `assert x_pred.shape == (4, 1)`.

**`actualizar(self, y_t: np.ndarray) -> tuple[np.ndarray, np.ndarray]`**
- Si `y_t` contiene NaN → omitir actualización, propagar predicción (EARS-K03).
- Ganancia: `K = P_{t|t-1}·C^T·(C·P_{t|t-1}·C^T + R)^{-1}`.
- `x̂_t = x̂_{t|t-1} + K·(y_t - C·x̂_{t|t-1})`.
- `P_t = (I - K·C)·P_{t|t-1}`.
- Precondición: `assert y_t.shape == (2, 1)`.

**`_simetrizar_covarianza(self, P: np.ndarray) -> np.ndarray`**
- Si `np.linalg.eigvals(P).min() < 0` → aplica `P = (P + P.T) / 2` y loguea advertencia (EARS-K05).
- Precondición: `assert P.shape == (4, 4)`.

**`paso(self, u_t: np.ndarray, y_t: np.ndarray) -> tuple[np.ndarray, np.ndarray]`**
- Llama `predecir` → `actualizar` → `_simetrizar_covarianza`.
- Actualiza `self.x_hat` y `self.P`.
- Retorna `(x_hat, P)`.

**`ejecutar_secuencia(self, U: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]`**
- Itera sobre `T` pasos, acumula `X_hat ∈ ℝ^{T×4}` y `P_trace ∈ ℝ^T` (traza de P en cada paso).
- Precondición: `assert U.shape[0] == Y.shape[0]`.

#### Parámetros hiperparámetros
- `Q = diag(0.01, 0.01, 0.01, 0.01)` (configurable).
- `R = diag(0.1, 0.1)` (configurable).
- Exponer como argumentos con defaults en `__init__`.

---

### 2.3 — Controlador LQR

**Archivo:** `modelos/dinamico/controlador.py`  
**EARS:** C01–C06

#### Descripción funcional
Calcula la ganancia de realimentación `K` resolviendo la ecuación de Riccati discreta. Produce la decisión de crédito `u_b = -K·x̂_c`, saturada en `[0, credit_limit_max]`. Expone `score_dinamico` normalizado en `[0, 1]`.

#### Funciones a implementar

**`calcular_ganancia_lqr(A: np.ndarray, B: np.ndarray, Q_lqr: np.ndarray, R_lqr: np.ndarray) -> np.ndarray`**
- Resuelve DARE con `scipy.linalg.solve_discrete_are(A, B, Q_lqr, R_lqr)`.
- `K = (R_lqr + B^T·P·B)^{-1}·B^T·P·A`.
- Precondición: `assert A.shape == (4, 4) and B.shape == (4, 1)`.
- Postcondición: `assert K.shape == (1, 4)`.

**`decidir_limite(K: np.ndarray, x_hat: np.ndarray, credit_limit_max: float) -> float`**
- `u_b = float(-K @ x_hat)`.
- Saturar: `limit_recomendado = max(0.0, min(credit_limit_max, u_b))`.
- Precondición: `assert x_hat.shape == (4, 1)`.
- Postcondición: `assert 0 <= limit_recomendado <= credit_limit_max`.

**`score_dinamico(x_hat: np.ndarray, P: np.ndarray, K: np.ndarray, credit_limit_max: float) -> float`**
- Combina el límite recomendado y la incertidumbre (traza de P) para producir score en `[0, 1]`.
- Fórmula orientativa: `score = limit_recomendado / credit_limit_max * (1 - lambda * trace(P))`, donde `lambda` es factor de penalización de incertidumbre configurable.
- Clip final: `max(0.0, min(1.0, score))`.
- Precondición: `assert 0 < credit_limit_max`.
- Postcondición: `assert 0.0 <= score <= 1.0`.

**`matrices_costo_default() -> tuple[np.ndarray, np.ndarray]`**
- Retorna `Q_lqr` y `R_lqr` con valores por defecto.
- `Q_lqr = diag(1, 1, 1, 10)` — penaliza fuertemente `days_in_default` (índice [3,3]).
- `R_lqr = np.array([[0.1]])`.
- Documenta que Q_lqr[3,3] alto → decisiones conservadoras (EARS-C04).

---

## Infraestructura SQLite y repositorios

**Archivos:** `gui/infraestructura/db.py`, `gui/infraestructura/repositorios/*.py`  
**EARS:** I01–I04

### Esquema de base de datos

#### Tabla `customers`
```sql
CREATE TABLE IF NOT EXISTS customers (
    customer_id TEXT PRIMARY KEY,
    archetype   TEXT NOT NULL
);
```

#### Tabla `monthly_states`
```sql
CREATE TABLE IF NOT EXISTS monthly_states (
    customer_id       TEXT,
    month             INTEGER,
    income            REAL,
    credit_limit      REAL,
    utilization_rate  REAL,
    outstanding_debt  REAL,
    payment_amount    REAL,
    days_in_default   INTEGER,
    num_transactions  INTEGER,
    transaction_vol   REAL,
    default_indicator INTEGER,
    PRIMARY KEY (customer_id, month)
);
```

#### Tabla `estimated_states`
```sql
CREATE TABLE IF NOT EXISTS estimated_states (
    customer_id    TEXT,
    month          INTEGER,
    x_hat_debt     REAL,
    x_hat_income   REAL,
    x_hat_util     REAL,
    x_hat_days     REAL,
    p_trace        REAL,
    score_dinamico REAL,
    PRIMARY KEY (customer_id, month)
);
```

#### Tabla `decisions`
```sql
CREATE TABLE IF NOT EXISTS decisions (
    customer_id       TEXT,
    month             INTEGER,
    limit_recomendado REAL,
    score_dinamico    REAL,
    score_logistico   REAL,
    PRIMARY KEY (customer_id, month)
);
```

#### Tabla `metrics`
```sql
CREATE TABLE IF NOT EXISTS metrics (
    modelo TEXT,
    metrica TEXT,
    valor  REAL,
    mes    INTEGER,
    PRIMARY KEY (modelo, metrica, mes)
);
```

### Funciones en `db.py`

**`inicializar_db(ruta: Path) -> sqlite3.Connection`**
- Crea el archivo si no existe (EARS-I04).
- Ejecuta `CREATE TABLE IF NOT EXISTS` para las 5 tablas.
- Precondición: `assert ruta.parent.exists()`.
- Postcondición: `assert ruta.exists()`.

**`obtener_conexion(ruta: Path) -> sqlite3.Connection`**
- Retorna conexión con `row_factory = sqlite3.Row`.
- Precondición: `assert ruta.exists()`.

### Repositorios

#### `cliente_repo.py` — `CustomerRepository`

**`insertar_cliente(conn, customer_id: str, archetype: str) -> None`**  
**`obtener_clientes(conn) -> list[dict]`**  
**`obtener_cliente(conn, customer_id: str) -> dict | None`**

#### `estado_repo.py` — `EstadoRepository`

**`insertar_estado_mensual(conn, estado: dict) -> None`**  
**`insertar_estado_estimado(conn, estimado: dict) -> None`** — guarda `x_hat` (4 floats), `p_trace`, `score_dinamico` (EARS-I02).  
**`obtener_trayectoria(conn, customer_id: str) -> list[dict]`**  
**`obtener_estado_estimado(conn, customer_id: str) -> list[dict]`**

#### `decision_repo.py` — `DecisionRepository`

**`insertar_decision(conn, decision: dict) -> None`**  
**`obtener_decisiones(conn, customer_id: str) -> list[dict]`**  
**`obtener_ultimas_n(conn, customer_id: str, n: int) -> list[dict]`**

---

## Fase 3 — Backtesting y comparación

**Archivo:** `modelos/evaluacion/backtesting.py`  
**Salidas:** `db/comparacion_modelos.json` + tablas SQLite pobladas  
**EARS:** V01–V06

### Prerequisito de Fase 3
Verificar existencia de `db/modelo_logistico.pkl` y `db/matrices_sistema.npz`. Si falta alguno → `FileNotFoundError` (EARS-DEP02).

### Descripción funcional
Simulación mes a mes de los 500 clientes bajo dos políticas:
1. **Modelo logístico**: puntúa cada mes con P(default); aplica una política simple de crédito basada en umbral.
2. **Modelo dinámico**: usa Kalman + LQR para decidir límite mes a mes.

Ambas simulaciones calculan pérdidas cuando hay default y se comparan al final.

### Funciones a implementar

**`calcular_perdida_mes(outstanding_debt: float, default: int, tasa_recuperacion: float = 0.30) -> float`**
- Si `default == 1` → `perdida = outstanding_debt * (1 - tasa_recuperacion)`.
- Si `default == 0` → `perdida = 0.0`.
- Precondición: `assert 0 <= tasa_recuperacion <= 1`.

**`simular_modelo_logistico(df_features: pd.DataFrame, modelo: Pipeline) -> pd.DataFrame`**
- Para cada `(customer_id, month)`, calcula P(default) y pérdida.
- Retorna DataFrame con columnas: `customer_id`, `month`, `prob_default`, `perdida`.
- Precondición: `assert 'default_indicator' in df_features.columns`.

**`simular_modelo_dinamico(df: pd.DataFrame, A, B, C, K, Q_k, R_k) -> pd.DataFrame`**
- Para cada cliente, instancia `FiltroKalman` y ejecuta Kalman + LQR mes a mes.
- Registra en cada paso: `x_hat`, `p_trace`, `score_dinamico`, `limit_recomendado`, `perdida`.
- Retorna DataFrame con esas columnas más `customer_id` y `month`.

**`calcular_gini_ks_auc(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]`**
- Reutiliza lógica de `_calcular_metricas` del módulo logístico.
- Precondición: `assert len(y_true) == len(y_score)`.

**`calcular_psi(score_base: np.ndarray, score_mes: np.ndarray, bins: int = 10) -> float`**
- PSI = Σ (Actual% − Esperado%) × ln(Actual% / Esperado%).
- Precondición: `assert len(score_base) > 0 and len(score_mes) > 0`.

**`calcular_psi_serie(scores_por_mes: dict[int, np.ndarray]) -> dict[int, float]`**
- Calcula PSI usando mes 1 como base para todos los meses posteriores.
- Retorna `{month: psi_value}`.

**`comparar_modelos(resultados_logistico, resultados_dinamico, df_features) -> dict`**
- Computa métricas para ambos modelos (Gini, KS, AUC, pérdida total, PSI mensual).
- Si pérdida dinámica no reduce ≥ 5% vs baseline → `warnings.warn(...)` y log (EARS-V05).
- Retorna dict para serialización JSON.

**`guardar_comparacion(comparacion: dict, ruta: Path) -> None`**
- Serializa con `json.dump`.
- Precondición: `assert ruta.parent.exists()`.

**`poblar_sqlite(conn, resultados_dinamico: pd.DataFrame, decisiones: pd.DataFrame) -> None`**
- Inserta filas en `estimated_states` y `decisions` vía repositorios.

**`ejecutar_backtesting(ruta_features: Path, ruta_matrices: Path, ruta_modelo_pkl: Path, ruta_salida_json: Path, conn) -> None`**
- Orquestador completo. Verifica prerequisitos, ejecuta ambas simulaciones, compara, guarda JSON y puebla SQLite.

---

## Fase 4 — Interfaz gráfica NiceGUI

**EARS:** GUI01–GUI05, DC01–DC06, DB01–DB07

### 4.0 — CSS global y estructura base

**Archivo:** `gui/global.css`

```css
/* Reset, variables de fuente, base font-size */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html { font-size: 10px; }
body { font-family: 'Inter', sans-serif; }
```

**Archivos:** `gui/cliente/styles.css`, `gui/banco/styles.css`  
- Variables de color para cada parte.
- **Cliente**: `--color-1: #3d8a5c` ... `--color-5: #f5faf5`.
- **Banco**: `--color-1: #2e3f52` ... `--color-5: #f4e1a4`.
- Todas las medidas en `rem`.

**Estrategia de inyección CSS en NiceGUI:**
- En el arranque de la aplicación, inyectar con `ui.add_head_html('<link rel="stylesheet" href="/gui/global.css">')`.
- Por ruta, inyectar el CSS local correspondiente.
- Usar clases prefijadas: `.cliente-score-gauge`, `.banco-panel-metricas`, etc.

### 4.1 — Arranque de la aplicación

**Archivo:** `main.py` (raíz del proyecto)

- Inicializa SQLite (si no existe).
- Verifica prerequisitos de todas las fases; ejecuta pipelines faltantes en orden.
- Registra rutas NiceGUI: `/cliente` y `/banco`.
- Llama `ui.run(...)`.

### 4.2 — Dashboard Cliente

**Ruta:** `/cliente`  
**Archivos:** `gui/cliente/paginas/dashboard_cliente.py` + componentes

#### Componentes

**`selector_cliente.py` — `SelectorCliente`**
- `ui.select` con búsqueda, poblado desde `CustomerRepository.obtener_clientes()`.
- Al cambiar → emite evento que actualiza todos los demás componentes.
- CSS class: `.cliente-selector`.

**`score_gauge.py` — `ScoreGauge`**
- Muestra score dinámico actual (0–100) como gauge circular.
- Implementado con SVG inline o `ui.echart` (gauge chart).
- Si `score < 0.40` → fondo del panel cambia a `#f5e6a4` (EARS-DC03).
- CSS class: `.cliente-score-gauge`.

**`trayectoria_chart.py` — `TrayectoriaChart`**
- Gráfico de líneas con las 4 dimensiones del estado estimado en subgráficos sincronizados.
- Implementado con `ui.echart` (4 series en subgráficos o 4 gráficas apiladas).
- Si hay meses con `default_indicator = 1` → marcador visual (punto rojo) en la línea (EARS-DC06).
- CSS class: `.cliente-trayectoria-chart`.

**`tabla_transacciones.py` — `TablaTransacciones`**
- Muestra últimas 6 filas con columnas: `mes`, `pago`, `deuda`, `estado estimado`.
- Implementado con `ui.table`.
- CSS class: `.cliente-tabla-transacciones`.

**`dashboard_cliente.py`**
- Layout: grid de 2 columnas — izquierda (selector + gauge), derecha (trayectoria + tabla).
- Maneja callbacks de `SelectorCliente` para actualizar los 3 componentes.

### 4.3 — Dashboard Banco

**Ruta:** `/banco`  
**Archivos:** `gui/banco/paginas/dashboard_banco.py` + componentes

#### Componentes

**`filtro_arquetipo.py` — `FiltroArquetipo`**
- `ui.select` o `ui.toggle` con valores: `good`, `recurrent`, `over`, `fraud`, `low`, `todos`.
- Filtra la cartera visible en los demás componentes (EARS-DB06).
- CSS class: `.banco-filtro-arquetipo`.

**`features_agregados.py` — `FeaturesAgregados`**
- Panel con 3 métricas agregadas de la cartera filtrada:
  - Ratio deuda/ingreso promedio.
  - Tendencia de utilización media.
  - Distribución de volatilidad de pagos (histograma pequeño con `ui.echart`).
- CSS class: `.banco-features-agregados`.

**`panel_metricas.py` — `PanelMetricas`**
- Tabla side-by-side: columnas `Métrica | Logístico | Dinámico`.
- Filas: Gini, KS, AUC, Pérdida Esperada.
- Carga datos desde `db/comparacion_modelos.json`.
- CSS class: `.banco-panel-metricas`.

**`autovalores_plot.py` — `AutovaloresPlot`**
- Plano complejo con autovalores de A ploteados como puntos.
- Círculo unitario superpuesto para visualizar estabilidad.
- Ganancia K como gráfico de barras horizontal.
- Implementado con `ui.echart` (scatter + círculo paramétrico).
- CSS class: `.banco-autovalores-plot`.

**`psi_chart.py` — `PSIChart`**
- Gráfico de línea del PSI mensual.
- Línea de umbral en `PSI = 0.25`.
- Puntos por encima del umbral marcados con alerta roja (EARS-DB05).
- CSS class: `.banco-psi-chart`.

**`simulacion_contrafactual.py` — `SimulacionContrafactual`**
- Selector de cliente.
- Dos series en el mismo gráfico: trayectoria con LQR vs. sin LQR (crédito fijo).
- Variable a visualizar seleccionable: `deuda`, `ingreso`, `utilizacion`, `dias_mora`.
- CSS class: `.banco-simulacion-contrafactual`.

**`dashboard_banco.py`**
- Layout de 3 secciones:
  1. **Cabecera**: filtro de arquetipo.
  2. **Fila 1**: features agregados (izq) + panel métricas (der).
  3. **Fila 2**: autovalores + PSI.
  4. **Fila 3**: simulación contrafactual (ancho completo).

---

## Tests pytest

**Archivos:** `tests/conftest.py` + archivos por módulo  
**EARS:** T01–T04

### `conftest.py`
- Fixtures reutilizables: `df_raw`, `df_features`, `A_mat`, `B_mat`, `C_mat`, `kalman_instance`, `modelo_logistico`.
- Fixture `db_tmp`: base de datos SQLite temporal en `tmp_path`.
- `SEED = 42` en fixture de datos sintéticos.

### `test_features.py`
- Verifica que `ratio_deuda_ingreso_ma` tenga ventana de 3 meses correcta.
- Verifica que `tendencia_utilizacion` sea 0.0 cuando solo hay 1 punto.
- Verifica que `income` NaN sea imputado antes de calcular ratio.
- Verifica que `customer_id` y `month` no cambien tras el pipeline.
- Verifica que `db/features_dinamicos.csv` se crea con las columnas esperadas.

### `test_kalman.py`
- Verifica que tras `paso()` el estado `x_hat` tenga forma `(4, 1)`.
- Verifica que si `y_t` contiene NaN, el estado se propaga sin actualización (solo predicción).
- Verifica que `P` permanece semidefinida positiva tras varios pasos.
- **Si `P` tiene autovalores negativos → el test FALLA con mensaje descriptivo** (EARS-T04).
- Verifica convergencia: tras 20 pasos con sistema estable, `trace(P)` decrece.

### `test_controlador.py`
- Verifica que `K` tiene forma `(1, 4)`.
- Verifica que `limit_recomendado` está en `[0, credit_limit_max]`.
- Verifica que con `Q_lqr[3,3]` grande el límite recomendado sea menor que con `Q_lqr[3,3]` pequeño.
- Verifica que `score_dinamico` ∈ `[0, 1]`.

### `test_logistico.py`
- Verifica que `predict_proba` retorna float en `[0, 1]`.
- Verifica que el modelo no se reentrena si el `.pkl` ya existe.
- Verifica que métricas `auc`, `gini`, `ks` están en rangos válidos.
- Verifica que el pipeline incluye `scaler` y `clf`.

### `test_integracion.py` (EARS-T03)
- Ejecuta el pipeline completo: generación → features → matrices → Kalman → LQR → backtesting.
- Verifica que `db/comparacion_modelos.json` existe al final.
- Verifica que la tabla `decisions` de SQLite tiene exactamente `500 × 24 = 12 000` filas.
- Verifica métricas mínimas: AUC logístico > 0.6, pérdida dinámica calculada.
- El test puede tardar; usar `@pytest.mark.slow` para poder excluirlo con `-m "not slow"`.

---

## Mapa de dependencias entre artefactos

```
scripts/transacciones.py
    → db/raw_transactions.csv

modelos/features/pipeline.py  ← db/raw_transactions.csv
    → db/features_dinamicos.csv

modelos/estatico/logistico.py  ← db/features_dinamicos.csv
    → db/modelo_logistico.pkl
    → db/metricas_baseline.json

modelos/dinamico/identificacion.py  ← db/features_dinamicos.csv
    → db/matrices_sistema.npz

modelos/dinamico/kalman.py  ← db/matrices_sistema.npz
    (no escribe archivo; instanciado en runtime)

modelos/dinamico/controlador.py  ← matrices_sistema.npz (A, B)
    (no escribe archivo; instanciado en runtime)

gui/infraestructura/db.py
    → db/credito.db (esquema)

modelos/evaluacion/backtesting.py
    ← db/features_dinamicos.csv
    ← db/modelo_logistico.pkl
    ← db/matrices_sistema.npz
    ← db/credito.db
    → db/comparacion_modelos.json
    → db/credito.db (pobla tablas)

gui/ (todas las rutas)
    ← db/credito.db
    ← db/comparacion_modelos.json
    ← db/metricas_baseline.json
```

---

## Criterios de aceptación globales

Un agente puede marcar el proyecto como **completado** cuando se cumplan **todos** los siguientes criterios:

| # | Criterio | Verificación |
|---|---|---|
| 1 | `scripts/transacciones.py --force` genera CSV con 12 000 filas | `wc -l db/raw_transactions.csv` = 12 001 |
| 2 | `db/features_dinamicos.csv` contiene las 3 columnas de features nuevas | `head -1 db/features_dinamicos.csv` |
| 3 | `db/modelo_logistico.pkl` existe y AUC > 0.6 | `db/metricas_baseline.json` |
| 4 | `db/matrices_sistema.npz` tiene claves A, B, C | `np.load('db/matrices_sistema.npz').files` |
| 5 | MSE de reconstrucción < 0.05 | Log de `identificacion.py` |
| 6 | Filtro de Kalman converge sin autovalores negativos | `pytest tests/test_kalman.py` |
| 7 | Score dinámico ∈ [0, 1] para todos los clientes | `pytest tests/test_controlador.py` |
| 8 | `db/comparacion_modelos.json` contiene Gini, KS, AUC, pérdida de ambos modelos | Inspección JSON |
| 9 | `pytest tests/` pasa con cobertura ≥ 80% en `modelos/` | `pytest --cov=modelos` |
| 10 | Dashboard cliente accesible en `/cliente` con selector, gauge, gráfico, tabla | Inspección visual |
| 11 | Dashboard banco accesible en `/banco` con panel métricas, PSI, autovalores, contrafactual | Inspección visual |
| 12 | Ninguna función supera 60 líneas | `grep -rn "def " modelos/ gui/` + conteo manual |
| 13 | Toda variable tiene tipo explícito | Revisión de código |
| 14 | Mínimo 2 aserciones por función | Revisión de código |
| 15 | `docs/tareas.md` refleja estado actualizado de todas las tareas | Inspección |

---

*Fin del plan de implementación.*