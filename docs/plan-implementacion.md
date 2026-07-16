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
8. [Infraestructura SQLite y repositorios](#infraestructura-sqlite-y-repositorios)
9. [Fase 3 — Backtesting y comparación](#fase-3--backtesting-y-comparación)
10. [Fase 4 — Revisión de código, patrones de diseño y logging](#fase-4--revisión-de-código-patrones-de-diseño-y-logging)
11. [Fase 5 — Interfaz gráfica Streamlit](#fase-5--interfaz-gráfica-streamlit)
12. [Tests pytest](#tests-pytest)
13. [Mapa de dependencias entre artefactos](#mapa-de-dependencias-entre-artefactos)
14. [Criterios de aceptación globales](#criterios-de-aceptación-globales)

---

## 1. Visión general del sistema

El sistema es una **aplicación monolítica Python 3.12** que construye un motor de scoring crediticio dinámico. En lugar de producir probabilidades estáticas, modela al cliente como un sistema dinámico con retroalimentación (banco → cliente → banco), estimando estado con Kalman y optimizando decisiones con LQR.

La interfaz gráfica es **Streamlit** con cuatro pestañas evaluando un **dataframe de evaluación nuevo, almacenado en RAM** (`st.session_state`), generado con semilla distinta a la de entrenamiento y transformado con el mismo pipeline de features.

### Flujo macro del sistema

```
Fase entrenamiento (offline, una vez):
  Generar datos sintéticos (SEED=42) → Feature engineering →
  Entrenar modelo logístico → Identificar matrices A, B, C →
  Backtesting → Poblar SQLite → Guardar JSON de comparación

Fase evaluación (Streamlit, en tiempo de ejecución):
  Generar dataset de evaluación (SEED=43) → Aplicar mismo pipeline →
  Cargar en st.session_state → 4 pestañas evalúan ambos modelos
```

### Componentes macro

| Componente | Responsabilidad |
|---|---|
| `scripts/transacciones.py` | Generación de datos sintéticos via `CustomerDirector` + `CustomerBuilder` |
| `modelos/dominio/` | Entidades de dominio: `CustomerBuilder`, `CustomerDirector`, arquetipos |
| `modelos/features/pipeline.py` | Feature engineering temporal (reutilizado en training y evaluación) |
| `modelos/estatico/logistico.py` | Baseline logístico, serialización en `.pkl` |
| `modelos/dinamico/identificacion.py` | Estimación de matrices A, B, C del sistema |
| `modelos/dinamico/kalman.py` | Filtro de Kalman discreto |
| `modelos/dinamico/controlador.py` | LQR, score dinámico normalizado |
| `modelos/evaluacion/backtesting.py` | Simulación mes a mes, comparación de modelos |
| `gui/infraestructura/db.py` | Inicialización SQLite, tablas |
| `gui/infraestructura/repositorios/` | Repositorios DDD (Customer, Decision, Estado) |
| `gui/paginas/` | Cuatro pestañas Streamlit |
| `gui/componentes/` | Componentes reutilizables de visualización |
| `gui/app.py` | Punto de entrada Streamlit: carga modelos, genera eval_df, renderiza pestañas |
| `utils/logger.py` | Singleton de logging con formato canónico |
| `utils/file_helpers.py` | Utilidades genéricas de I/O |
| `settings/settings.py` | Clase con configuraciones globales para constantes |

### Vector de estado

El estado del cliente es **ℝ³**: `x_c = [outstanding_debt, income, utilization_rate]ᵀ`.

| Símbolo | Dimensión | Variables |
|---|---|---|
| `x_c` | ℝ³ | `outstanding_debt`, `income`, `utilization_rate` |
| `u_t` | ℝ¹ | `credit_limit` |
| `y_t` | ℝ² | `num_transactions`, `payment_amount` |

---

## 2. Restricciones y reglas de codificación

Estas reglas aplican a **cada función de cada archivo** y deben verificarse antes de marcar una tarea como completada:

| Regla | Descripción |
|---|---|
| **Longitud** | Ninguna función supera 60 líneas |
| **Tipado explícito** | Toda variable y parámetro lleva anotación de tipo |
| **Aserciones** | Mínimo 2 aserciones por función (precondiciones / invariantes) |
| **Legibilidad** | Código autoexplicativo; mínimos comentarios inline |
| **Paradigma** | POO > funcional > procedural |
| **DRY** | Sin duplicación; reutilizar implementaciones existentes |
| **DDD** | Capas dominio / aplicación / infraestructura desacopladas |
| **Seed fijo** | `SEED = 42` para entrenamiento, `SEED = 43` para evaluación |
| **Singleton .pkl** | Modelo logístico entrenado una sola vez; reutilizado en adelante |
| **FileNotFoundError** | Si falta dependencia de fase, lanzar con ruta y fase |
| **Logging** | Toda función relevante registra entrada / salida / errores mediante `utils/logger.py` |
| **Builder obligatorio** | Creación de clientes y trayectorias **siempre** via `CustomerBuilder` + `CustomerDirector` |
| **Streamlit state** | El dataframe de evaluación vive en `st.session_state['eval_df']`; nunca se recalcula si ya existe |

---

## 3. Estructura de archivos objetivo

```
dashboard-dinamic-score-system/
├── AGENTS.md
├── main.py                             # Punto de entrada: streamlit run main.py
├── requirements.txt
├── db/
│   ├── credito.db
│   ├── raw_transactions.csv            # Training SEED=42
│   ├── eval_transactions.csv           # Evaluation SEED=43
│   ├── features_dinamicos.csv          # Training features
│   ├── eval_features.csv               # Evaluation features (en disco, cargado en RAM)
│   ├── matrices_sistema.npz
│   ├── matrices_sistema.json           # Scale params de normalización
│   ├── modelo_logistico.pkl
│   ├── metricas_baseline.json
│   └── comparacion_modelos.json
├── modelos/
│   ├── __init__.py
│   ├── dominio/
│   │   ├── __init__.py
│   │   ├── arquetipos.py               # Constantes y configuración por arquetipo
│   │   ├── customer_builder.py         # Patrón Builder: construye trayectoria de 1 cliente
│   │   └── customer_director.py        # Director: orquesta composición de 500 clientes
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
│   ├── __init__.py
│   ├── app.py                          # Bootstrap de Streamlit, carga recursos, renderiza tabs
│   ├── infraestructura/
│   │   ├── __init__.py
│   │   ├── db.py
│   │   └── repositorios/
│   │       ├── __init__.py
│   │       ├── cliente_repo.py
│   │       ├── decision_repo.py
│   │       └── estado_repo.py
│   ├── paginas/
│   │   ├── __init__.py
│   │   ├── analisis_descriptivo.py
│   │   ├── modelo_estatico.py
│   │   ├── modelo_dinamico.py
│   │   └── comparacion.py
│   └── componentes/
│       ├── __init__.py
│       ├── metricas_card.py
│       ├── graficos_roc.py
│       ├── graficos_kalman.py
│       ├── graficos_comparacion.py
│       └── shock_simulator.py
├── settings/
│   └── settings.py
├── utils/
│   ├── __init__.py
│   ├── logger.py                       # Singleton de logging con formato canónico
│   ├── file_helpers.py                 # I/O genérico
│   └── transacciones.py                # CLI wrapper: usa CustomerDirector
├── tests/
│   ├── conftest.py
│   ├── test_features.py
│   ├── test_kalman.py
│   ├── test_controlador.py
│   ├── test_logistico.py
│   ├── test_builder.py
│   └── test_integracion.py
└── docs/
    ├── plan-implementacion.md
    └── tareas.md
```

> **Nota de corrección:** El directorio `modelos/evaulacion/` (con typo) debe ser renombrado a `modelos/evaluacion/` durante la Fase 4. Todos los imports deben actualizarse en consecuencia.

---

## 4. Orden de implementación global

```
Fase 0 (setup)
    ↓
Fase 1.1 (CustomerBuilder + CustomerDirector → raw_transactions.csv + eval_transactions.csv)
    ↓
Fase 1.2 (pipeline.py → features_dinamicos.csv + eval_features.csv)
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
Fase 4 (revisión de código: Builder, logging, DRY, corrección typo evaluacion)
    ↓
Fase 5 (Streamlit: gui/app.py + 4 páginas + componentes)
    ↓
Tests (conftest → unit tests → test_builder → test integración)
```

> Las Fases 0–3 e Infraestructura están **semi-completadas** (tienen código inicial, pero requiere mejorarlos siguiendo el plan de implementación). El trabajo pendiente comienza en Fase 0.

---

## Fase 0 — Bootstrapping del proyecto

**Estado:** ✅ Completada

### Tareas completadas

- T0.1 Estructura de directorios y `__init__.py` vacíos.
- T0.2 `requirements.txt` con dependencias fijadas.
- T0.3 `docs/tareas.md` y `docs/plan-implementacion.md`.
- T0.4 Validación Python 3.12, instalación de dependencias.

### Pendiente de Fase 0 (añadido)

#### T0.5 — Crear `utils/` y `modelos/dominio/`
- Crear directorios `utils/` y `modelos/dominio/`.
- Crear `utils/__init__.py`, `modelos/dominio/__init__.py`.
- Crear archivos vacíos: `utils/logger.py`, `utils/file_helpers.py`, `modelos/dominio/arquetipos.py`, `modelos/dominio/customer_builder.py`, `modelos/dominio/customer_director.py`.

#### T0.6 — Crear estructura `gui/paginas/` y `gui/componentes/`
- Crear `gui/paginas/` con `__init__.py` y archivos vacíos de las cuatro páginas.
- Crear `gui/componentes/` con `__init__.py` y archivos vacíos de cada componente.
- Crear `gui/app.py` vacío.
- Crear `main.py` vacío en raíz.

#### T0.7 — Actualizar `requirements.txt`
- Agregar `streamlit>=1.35` si no está presente.
- Verificar que `nicegui` puede ser eliminado si ya no se utiliza.

---

## Fase 1 — Dataset y features dinámicos

### 1.0 — Dominio: Builder de clientes

**Archivos:** `modelos/dominio/arquetipos.py`, `modelos/dominio/customer_builder.py`, `modelos/dominio/customer_director.py`

> Esta fase refactoriza la lógica de generación de datos de `scripts/transacciones.py` hacia el patrón Builder, cumpliendo EARS-F5-01.

#### `modelos/dominio/arquetipos.py`

Define la configuración de comportamiento de cada arquetipo como constantes o dataclasses:

**`ARCHETYPE_PARAMS: dict[str, dict]`**
- Diccionario con claves `good`, `recurrent`, `over`, `fraud`, `low`.
- Cada entrada contiene los parámetros de distribución usados en la generación de filas (utilization media/std, default_prob, income_shock_prob, etc.).
- Precondición: `assert set(ARCHETYPE_PARAMS.keys()) == {"good", "recurrent", "over", "fraud", "low"}`.

**`ARCHETYPE_COMPOSITION: dict[str, float]`**
- Proporciones: `good=0.40`, `recurrent=0.25`, `over=0.20`, `fraud=0.05`, `low=0.10`.
- Precondición: `assert abs(sum(ARCHETYPE_COMPOSITION.values()) - 1.0) < 1e-9`.

#### `modelos/dominio/customer_builder.py` — clase `CustomerBuilder`

Construye la trayectoria mensual completa de **un único cliente** (24 filas).

**`__init__(self)`**
- Inicializa atributos internos: `_customer_id`, `_archetype`, `_base_income`, `_credit_limit` (todos `None`).
- Aserciones: objeto creado en estado no construido.

**`with_customer_id(self, customer_id: str) -> 'CustomerBuilder'`**
- Precondición: `assert customer_id.startswith('C')`.

**`with_archetype(self, archetype: str) -> 'CustomerBuilder'`**
- Precondición: `assert archetype in ARCHETYPE_PARAMS`.

**`with_base_income(self, income: float) -> 'CustomerBuilder'`**
- Precondición: `assert income > 0`.

**`with_credit_limit(self, limit: float) -> 'CustomerBuilder'`**
- Precondición: `assert limit > 0`.

**`build(self) -> list[dict]`**
- Verifica que los 4 atributos obligatorios están seteados.
- Delega la generación mes a mes a métodos privados `_build_month_<archetype>(month, ...)`.
- Retorna lista de 24 dicts, uno por mes.
- Postcondición: `assert len(resultado) == settings.MONTHS`.
- Postcondición: `assert all('customer_id' in r for r in resultado)`.

**Métodos privados `_build_month_good`, `_build_month_recurrent`, `_build_month_over`, `_build_month_fraud`, `_build_month_low`**
- Encapsulan la lógica de generación de filas del script actual.
- Cada uno acepta `(self, month: int, current_limit: float) -> tuple[dict, float]` (fila + nuevo límite).
- Deben incorporar los shocks exógenos (1–5% probabilidad) y drift temporal definidos en AGENTS.md.

#### `modelos/dominio/customer_director.py` — clase `CustomerDirector`

Orquesta la generación de la cartera completa usando `CustomerBuilder`.

**`__init__(self, builder: CustomerBuilder)`**
- Precondición: `assert isinstance(builder, CustomerBuilder)`.
- Guarda referencia al builder.

**`_calcular_composicion(self, n: int) -> dict[str, int]`**
- Calcula cantidad de clientes por arquetipo según `ARCHETYPE_COMPOSITION`.
- Precondición: `assert n == settings.N_CUSTOMERS`.
- Postcondición: `assert sum(composicion.values()) == n`.

**`construir_dataset(self, seed: int) -> pd.DataFrame`**
- Fija `np.random.seed(seed)`.
- Mezcla lista de arquetipos según composición.
- Itera sobre 500 clientes: asigna ID, income aleatorio, credit_limit; llama `builder.with_*(...).build()`.
- Apila resultados en DataFrame.
- Precondición: `assert seed > 0`.
- Postcondición: `assert len(df) == settings.N_CUSTOMERS * settings.MONTHS`.

---

### 1.1 — Generación de datos sintéticos

**Archivo:** `scripts/transacciones.py`
**Salidas:** `db/raw_transactions.csv` (SEED=42), `db/eval_transactions.csv` (SEED=43)

**Estado:** Parcialmente completada (lógica existente). Pendiente: refactorizar para usar `CustomerDirector`, generar dataset de evaluación.

#### T1.1.1 — Refactorizar `scripts/transacciones.py` para usar `CustomerDirector`

- Eliminar funciones `_row_good`, `_row_recurrent`, etc. del script; moverlas a `CustomerBuilder`.
- El script queda como wrapper CLI que:
  1. Parsea args: `--force`, `--seed` (default=42), `--output` (default=`settings.RAW_TRANSACTIONS_PATH`).
  2. Instancia `CustomerBuilder` + `CustomerDirector`.
  3. Llama `director.construir_dataset(seed=args.seed)`.
  4. Guarda CSV con `save_csv(df, path)`.
- Si el CSV ya existe y no se pasa `--force`, imprime mensaje y termina.

#### T1.1.2 — Generar dataset de evaluación

- Llamada adicional con `seed=43` → `db/eval_transactions.csv`.
- Puede hacerse desde `main.py` o añadirse como segunda ejecución del script.
- El dataset de evaluación representa **clientes nuevos**, no vistos en entrenamiento.

#### T1.1.3 — Verificación

- `db/raw_transactions.csv`: 12 000 filas, columnas correctas.
- `db/eval_transactions.csv`: 12 000 filas, misma estructura.

---

### 1.2 — Feature Engineering Dinámico

**Archivo:** `modelos/features/pipeline.py`
**Salidas:** `db/features_dinamicos.csv`, `db/eval_features.csv`

**Estado:** ✅ Completada para training. Pendiente: `generar_features` debe ser invocada también sobre `eval_transactions.csv`.

#### T1.2.1 — Generar eval features

- `generar_features(settings.EVAL_TRANSACTIONS_PATH, settings.EVAL_FEATURES_PATH)`.
- Aserciones existentes aplican igual.
- Llamada desde `main.py` si `eval_features.csv` no existe.

---

### 1.3 — Modelo Logístico Baseline

**Estado:** ✅ Completada. No requiere cambios funcionales. El eval dataset se evaluará en Fase 5, no en este módulo.

---

## Fase 2 — Modelo dinámico (Kalman + LQR)

**Estado:** ✅ Completada. Las implementaciones existentes en `identificacion.py`, `kalman.py` y `controlador.py` son correctas con `N_STATES=3`.

> **Recordatorio dimensional:** El vector de estado tiene **3 componentes** (`outstanding_debt`, `income`, `utilization_rate`). Las matrices Q_lqr, A, K, P son de dimensión 3×3, 3×3, 1×3, 3×3 respectivamente.

No se realizan cambios funcionales en Fase 2. Los ajustes de logging y patrones se incorporan en Fase 4.

---

## Infraestructura SQLite y repositorios

**Estado:** ✅ Completada. Archivos `db.py`, `cliente_repo.py`, `estado_repo.py`, `decision_repo.py` funcionan correctamente.

---

## Fase 3 — Backtesting y comparación

**Estado:** ✅ Completada. Salidas: `db/comparacion_modelos.json` + tablas SQLite pobladas.

---

## Fase 4 — Revisión de código, patrones de diseño y logging

**Prerequisito:** Fases 0–3 e Infraestructura completadas.

Esta fase aplica EARS-F5-01, EARS-F5-02, EARS-G04, EARS-G05, EARS-G06 sobre el código ya escrito, y añade el sistema de logging transversal.

---

### 4.0 — Corrección de typo crítico

#### T4.0.1 — Renombrar `modelos/evaulacion/` a `modelos/evaluacion/`

- Renombrar el directorio en disco.
- Actualizar todos los imports que referencian `modelos.evaulacion.*` → `modelos.evaluacion.*`.
- Verificar que `from modelos.evaluacion.backtesting import ...` no lanza `ModuleNotFoundError`.

---

### 4.1 — Sistema de logging canónico

**Archivo:** `utils/logger.py`

**Formato canónico:**
```
[TIMESTAMP] | [LEVEL] | [SERVICE / MODULE] | [TRACE ID] | MESSAGE
```

Ejemplo:
```
[2026-06-27 14:32:01] | INFO | modelos.dinamico.kalman | 7a3f | FiltroKalman.step: t=12, trace(P)=0.0423
```

#### Función `setup_logging(level: int = logging.INFO) -> None`

- Configura el logger raíz **una única vez** con `logging.basicConfig`.
- Formato: `[%(asctime)s] | %(levelname)s | %(name)s | %(process)d | %(message)s`.
- Verifica que no se configure dos veces usando un flag de módulo `_CONFIGURED: bool = False`.
- Precondición: `assert isinstance(level, int)`.
- Postcondición: `assert logging.getLogger().hasHandlers()`.

#### Función `get_logger(module_name: str) -> logging.Logger`

- Retorna `logging.getLogger(module_name)`.
- Precondición: `assert module_name.strip() != ''`.
- Todo módulo del proyecto llama `logger = get_logger(__name__)` al inicio del archivo.

#### `utils/file_helpers.py`

**`leer_csv(path: Path) -> pd.DataFrame`**
- Lee CSV con logging de entrada y forma del resultado.
- Precondición: `assert path.exists()`.
- Lanza `FileNotFoundError` con ruta si no existe.

**`leer_json(path: Path) -> dict`**
- Lee JSON con logging.
- Precondición: `assert path.suffix == '.json'`.

**`escribir_csv(df: pd.DataFrame, path: Path) -> None`**
- Escribe CSV con logging de filas escritas.
- Precondición: `assert len(df) > 0`.

---

### 4.2 — Integración de logging en módulos existentes

#### T4.2.1 — Agregar logging a `modelos/features/pipeline.py`

- Al inicio: `logger = get_logger(__name__)`.
- En `generar_features`: log inicio, log de cada función aplicada, log de filas en salida.
- En `imputar_income`: log si hubo NaNs imputados.

#### T4.2.2 — Agregar logging a `modelos/estatico/logistico.py`

- En `train_and_save`: log inicio de entrenamiento, log de métricas resultantes.
- En `initialize`: log si carga desde disco o entrena nuevo.
- En `predict_proba`: log de customer_id, month y probabilidad resultante a nivel DEBUG.

#### T4.2.3 — Agregar logging a `modelos/dinamico/identificacion.py`

- En `identify`: log si carga matrices o las recalcula.
- En `verify_mse`: log del MSE calculado (INFO si OK, WARNING si ≥ 0.05).
- En `identify_AB`: log de autovalores de A (WARNING si inestable).

#### T4.2.4 — Agregar logging a `modelos/dinamico/kalman.py`

- En `_symmetrize_covariance`: ya emite `warnings.warn`; añadir `logger.warning` con mismo mensaje.
- En `execute_sequence`: log resumen al final (T pasos, trace(P) final).

#### T4.2.5 — Agregar logging a `modelos/dinamico/controlador.py`

- En `calculate_lqr_gain`: log de norma de K resultante.
- En `dynamic_score`: log a nivel DEBUG de score y trace(P).

#### T4.2.6 — Agregar logging a `modelos/evaluacion/backtesting.py`

- En `run_backtesting`: log de cada etapa (carga modelo, simulación logístico, simulación dinámico, comparación, SQLite).
- En `compare_models`: log de reducción de pérdida y advertencia si < 5%.

#### T4.2.7 — Agregar logging a `gui/infraestructura/db.py`

- En `inicializar_db`: log de tablas creadas.
- En `obtener_conexion`: log de conexión establecida.

---

### 4.3 — Revisión de calidad de código

#### T4.3.1 — Auditoría de longitud de funciones

- Verificar que ninguna función supera 60 líneas.
- Si alguna supera el límite, extraer subfunciones con nombres descriptivos.
- Herramienta sugerida: `grep -rn "def " modelos/ scripts/ gui/ | wc -l` como punto de partida.

#### T4.3.2 — Auditoría de tipado explícito

- Toda variable local, parámetro y retorno debe tener anotación de tipo.
- Verificar con `mypy modelos/ utils/ gui/ scripts/` (ignorar errores de librerías externas).

#### T4.3.3 — Auditoría de aserciones

- Mínimo 2 aserciones por función.
- Revisar especialmente funciones de repositorios y utilidades.

#### T4.3.4 — Auditoría DRY

- Identificar lógica duplicada entre `backtesting.py` y `logistico.py` (cálculo de métricas).
- `_calculate_metrics` ya existe en `logistico.py` y se importa en `backtesting.py` — verificar que no hay copia local.

---

### 4.4 — Integración del Builder en `scripts/transacciones.py`

#### T4.4.1 — Refactorizar script para usar `CustomerDirector`

- El script importa `CustomerBuilder` y `CustomerDirector` de `modelos.dominio`.
- La función `main()` queda reducida a: parsear args → instanciar director → construir dataset → guardar CSV.
- Las funciones `_row_*` y `generate_customer` se eliminan del script.

#### T4.4.2 — Mover lógica de generación de filas al Builder

- Cada `_build_month_<archetype>` en `CustomerBuilder` recibe la lógica actualmente en `_row_*`.
- Los shocks exógenos (1–5%) deben estar incorporados en `arquetipos.py` como `SHOCK_PROB` por arquetipo.
- El drift temporal debe introducirse en el Builder según el mes y el arquetipo.

---

## Fase 5 — Interfaz gráfica Streamlit

**Prerequisito:** Fases 0–4 completadas, `db/eval_features.csv` existe.

**Archivo de entrada:** `main.py` (raíz) → `streamlit run main.py`.

---

### 5.0 — Bootstrap de la aplicación

**Archivo:** `main.py`

```
main.py:
  1. Llama setup_logging() de utils.logger
  2. Verifica y ejecuta pipelines faltantes en orden (EARS-G09):
     - raw_transactions.csv → pipeline.py → features_dinamicos.csv
     - eval_transactions.csv → pipeline.py → eval_features.csv
     - modelo_logistico.pkl (initialize)
     - matrices_sistema.npz (identificar)
     - comparacion_modelos.json (run_backtesting)
  3. Llama streamlit run gui/app.py (o importa y ejecuta gui.app.run())
```

> En Streamlit, `main.py` es el archivo pasado a `streamlit run`. Ejecuta `gui.app.run()`.

**Archivo:** `gui/app.py`

**Función `run() -> None`**:
- Configura `st.set_page_config(page_title="Credit Score Dashboard", layout="wide")`.
- Carga recursos en `st.session_state` si no existen (`@st.cache_resource` para modelos):
  - `modelo`: `load_model(settings.LOGISTICS_MODEL_PATH)`.
  - `A, B, C`: `load_matrices(settings.MATRIX_SYSTEM_PATH)`.
  - `K`: `calculate_lqr_gain(A, B, *default_cost_matrices())`.
  - `scale_params`: JSON de `settings.MATRIX_SYSTEM_PATH.with_suffix('.json')`.
  - `comparacion`: JSON de `settings.COMPARISON_PATH`.
- Carga `eval_df` en `st.session_state`:
  - Si `'eval_df'` no existe en session_state → `pd.read_csv(settings.EVAL_FEATURES_PATH)`.
  - Almacenar como `st.session_state['eval_df']`.
- Renderiza sidebar con descripción del proyecto y selector de pestaña.
- Renderiza pestañas con `st.tabs(["Análisis Descriptivo", "Modelo Estático", "Modelo Dinámico", "Comparación"])`.
- Delega cada pestaña a su módulo de página correspondiente.
- Precondición: `assert 'eval_df' in st.session_state`.
- Precondición: `assert not st.session_state['eval_df'].empty`.

---

### 5.1 — Pestaña 1: Análisis Descriptivo y Diagnóstico

**Archivo:** `gui/paginas/analisis_descriptivo.py`

#### Función `render(eval_df: pd.DataFrame) -> None`

Orquestador de la pestaña. Llama a los componentes en orden.

Precondición: `assert isinstance(eval_df, pd.DataFrame) and len(eval_df) > 0`.

#### `gui/componentes/metricas_card.py`

**`render_resumen_estadistico(df: pd.DataFrame) -> None`**
- `st.dataframe` con media, std, min, max, % de nulos por columna numérica.
- Precondición: `assert not df.empty`.
- Precondición: `assert len(df.select_dtypes('number').columns) > 0`.

**`render_distribucion_arquetipos(df: pd.DataFrame) -> None`**
- `st.bar_chart` o `st.plotly_chart` con conteo por `archetype` (si la columna existe).
- Precondición: `assert 'customer_id' in df.columns`.

**`render_alertas_diagnostico(df: pd.DataFrame, train_df: pd.DataFrame | None = None) -> None`**
- Calcula correlaciones entre columnas numéricas; alerta si `|r| > 0.9`.
- Detecta outliers: Z-score > 3 por columna; reporta conteo.
- Si `train_df` proporcionado: KS-test por columna entre eval y train; alerta si p-value < 0.05 (posible drift).
- Muestra alertas en `st.warning(...)` o `st.error(...)`.
- Precondición: `assert len(df) > 0`.
- Precondición: `assert df.select_dtypes('number').shape[1] >= 2`.

**`render_histogramas(df: pd.DataFrame, columnas: list[str]) -> None`**
- Renderiza histogramas en grid de 2 columnas usando `st.columns`.
- Precondición: `assert len(columnas) > 0`.
- Precondición: `assert all(c in df.columns for c in columnas)`.

---

### 5.2 — Pestaña 2: Modelo Estático

**Archivo:** `gui/paginas/modelo_estatico.py`

#### Función `render(eval_df: pd.DataFrame, modelo: Pipeline) -> None`

Precondición: `assert 'default_indicator' in eval_df.columns`.
Precondición: `assert hasattr(modelo, 'predict_proba')`.

Flujo interno:
1. Imputar `settings.FEATURE_COLUMNS` con `fillna(0.0)`.
2. Calcular `y_prob = modelo.predict_proba(eval_df[settings.FEATURE_COLUMNS])[:, 1]`.
3. Extraer `y_true = eval_df[settings.TARGET_COLUMN].values`.
4. Llamar componentes de visualización.

#### `gui/componentes/graficos_roc.py`

**`render_curva_roc(y_true: np.ndarray, y_prob: np.ndarray) -> None`**
- Calcula `fpr, tpr, _` con `roc_curve`.
- Plotea con `st.plotly_chart` (línea + diagonal de referencia).
- Muestra AUC en título o subtítulo.
- Precondición: `assert len(y_true) == len(y_prob)`.
- Precondición: `assert y_prob.min() >= 0.0 and y_prob.max() <= 1.0`.

**`render_ks_plot(y_true: np.ndarray, y_prob: np.ndarray) -> None`**
- Calcula distribuciones acumuladas de scores para clase 0 y clase 1.
- Plotea ambas curvas; marca distancia KS máxima.
- Muestra valor KS en subtítulo.
- Precondición: `assert len(np.unique(y_true)) == 2`.
- Precondición: `assert len(y_true) == len(y_prob)`.

**`render_metricas_tabla(metricas: dict[str, float]) -> None`**
- `st.metric` en 3 columnas: AUC, Gini, KS.
- Precondición: `assert {'auc', 'gini', 'ks'}.issubset(metricas.keys())`.
- Precondición: `assert all(0.0 <= v <= 1.0 for v in metricas.values())`.

**`render_psi_temporal(eval_df: pd.DataFrame, modelo: Pipeline) -> None`**
- Calcula score logístico por mes sobre `eval_df`.
- Calcula PSI del mes 1 como base respecto a meses 2–24.
- Plotea serie temporal de PSI con línea de umbral en 0.25.
- Precondición: `assert 'month' in eval_df.columns`.
- Precondición: `assert eval_df['month'].nunique() >= 2`.

**`render_scatter_limites(eval_df: pd.DataFrame, y_prob: np.ndarray) -> None`**
- Scatter: eje X = `credit_limit`, eje Y = probabilidad de default.
- Color por `default_indicator`.
- Precondición: `assert 'credit_limit' in eval_df.columns`.
- Precondición: `assert len(eval_df) == len(y_prob)`.

---

### 5.3 — Pestaña 3: Modelo Dinámico

**Archivo:** `gui/paginas/modelo_dinamico.py`

#### Función `render(eval_df: pd.DataFrame, A: np.ndarray, B: np.ndarray, C: np.ndarray, K: np.ndarray, scale_params: dict) -> None`

Precondición: `assert A.shape == (settings.N_STATES, settings.N_STATES)`.
Precondición: `assert 'customer_id' in eval_df.columns`.

Flujo:
1. Selector de cliente (`st.selectbox` con IDs únicos de `eval_df`).
2. Filtrar `cliente_df = eval_df[eval_df['customer_id'] == cliente_id]`.
3. Ejecutar Kalman+LQR sobre `cliente_df` (sin guardar en DB, solo en memoria).
4. Llamar componentes de visualización.

> La simulación en esta pestaña es **en memoria**, sin escritura a disco ni SQLite.

#### `gui/componentes/graficos_kalman.py`

**`run_kalman_for_customer(df: pd.DataFrame, A, B, C, K, scale_params: dict) -> pd.DataFrame`**
- Instancia `FiltroKalman` con `x0=zeros(3,1)`, `P0=eye(3)`.
- Itera mes a mes: normaliza observaciones, ejecuta `kalman.step`.
- Calcula `score_dinamico` y `limit_recomendado` por mes.
- Retorna DataFrame con columnas: `month`, `x_hat_debt`, `x_hat_income`, `x_hat_util`, `p_trace`, `score_dinamico`, `limit_recomendado`.
- Precondición: `assert len(df) > 0`.
- Precondición: `assert all(c in scale_params for c in settings.STATES)`.

**`render_estados_kalman(cliente_df: pd.DataFrame, kalman_df: pd.DataFrame) -> None`**
- Tres subplots apilados: `outstanding_debt`, `income`, `utilization_rate`.
- Cada subplot: valor real (línea continua) vs estado estimado x̂ (línea discontinua).
- Usa `st.plotly_chart` con subplots sincronizados en eje X (mes).
- Marca con punto rojo meses donde `default_indicator == 1`.
- Precondición: `assert 'month' in cliente_df.columns and 'month' in kalman_df.columns`.
- Precondición: `assert len(cliente_df) == len(kalman_df)`.

**`render_limite_credito_dinamico(kalman_df: pd.DataFrame, cliente_df: pd.DataFrame) -> None`**
- Gráfico de líneas: límite recomendado por LQR vs `credit_limit` real del eval_df.
- Precondición: `assert 'limit_recomendado' in kalman_df.columns`.
- Precondición: `assert 'credit_limit' in cliente_df.columns`.

**`render_convergencia_traza(kalman_df: pd.DataFrame) -> None`**
- Gráfico de línea de `p_trace` a lo largo de los 24 meses.
- Añade anotación si la traza decrece monótonamente (convergencia).
- Precondición: `assert 'p_trace' in kalman_df.columns`.
- Precondición: `assert len(kalman_df) > 1`.

---

### 5.4 — Pestaña 4: Comparación de modelos

**Archivo:** `gui/paginas/comparacion.py`

#### Función `render(eval_df: pd.DataFrame, modelo: Pipeline, A, B, C, K, scale_params: dict) -> None`

Precondición: `assert 'default_indicator' in eval_df.columns`.
Precondición: `assert K.shape == (settings.N_CONTROL, settings.N_STATES)`.

Flujo:
1. Simular modelo logístico sobre `eval_df` → `log_results`.
2. Simular modelo dinámico sobre `eval_df` → `dyn_results`.
3. Calcular métricas de comparación.
4. Llamar componentes de visualización.
5. Renderizar shock simulator al final.

> Las simulaciones son **en memoria** usando `simulate_logistic_model` y `simulate_dynamic_model` de `backtesting.py`. El resultado **no se persiste a SQLite**.

#### `gui/componentes/graficos_comparacion.py`

**`render_tabla_metricas_comparacion(log_metrics: dict, dyn_metrics: dict, reduccion: float) -> None`**
- Tabla con filas: AUC, Gini, KS, Pérdida Total.
- Columnas: Logístico | Dinámico | Δ.
- `st.metric` para reducción de pérdida con `delta` positivo o negativo.
- Precondición: `assert {'auc', 'gini', 'ks', 'perdida_total'}.issubset(log_metrics.keys())`.
- Precondición: `assert {'auc', 'gini', 'ks', 'perdida_total'}.issubset(dyn_metrics.keys())`.

**`render_perdida_acumulada_mensual(log_results: pd.DataFrame, dyn_results: pd.DataFrame) -> None`**
- Calcula pérdida acumulada mes a mes para ambos modelos.
- Gráfico de líneas doble: logístico vs dinámico.
- Precondición: `assert 'loss' in log_results.columns and 'loss' in dyn_results.columns`.
- Precondición: `assert 'month' in log_results.columns and 'month' in dyn_results.columns`.

**`render_tasa_mora_mensual(log_results: pd.DataFrame, dyn_results: pd.DataFrame) -> None`**
- Calcula % de clientes con `default_indicator == 1` por mes.
- Gráfico de líneas doble.
- Precondición: `assert 'default_indicator' in log_results.columns`.
- Precondición: `assert 'month' in log_results.columns`.

**`render_estabilidad_limites(dyn_results: pd.DataFrame) -> None`**
- Calcula desviación estándar de `limit_recomendado` por cliente a lo largo del tiempo.
- Histograma de estabilidad; cuanto menor la desviación, más estable la política.
- Precondición: `assert 'limit_recomendado' in dyn_results.columns`.
- Precondición: `assert 'customer_id' in dyn_results.columns`.

#### `gui/componentes/shock_simulator.py`

**`render_shock_simulator(eval_df: pd.DataFrame, modelo: Pipeline, A, B, C, K, scale_params: dict) -> None`**

Componente interactivo que recalcula el impacto de un shock macroeconómico.

Controles Streamlit:
- `st.slider("Incremento de deuda (%)", 0, 50, 0)` → `shock_debt`.
- `st.slider("Caída de ingresos (%)", 0, 40, 0)` → `shock_income`.
- `st.button("Aplicar shock")`.

**`_aplicar_shock(df: pd.DataFrame, shock_debt: float, shock_income: float) -> pd.DataFrame`**
- Crea copia de `df`.
- Multiplica `outstanding_debt` por `(1 + shock_debt / 100)`.
- Multiplica `income` por `(1 - shock_income / 100)`.
- Recalcula `utilization_rate = outstanding_debt / credit_limit`, saturado en [0, 1].
- Precondición: `assert 0 <= shock_debt <= 50`.
- Precondición: `assert 0 <= shock_income <= 40`.
- Postcondición: `assert len(df_shock) == len(df)`.

**Flujo al activar shock:**
1. Llama `_aplicar_shock` → `df_shock`.
2. Simula ambos modelos sobre `df_shock` → `log_shock`, `dyn_shock`.
3. Muestra delta de pérdida total: `Δ loss logístico` y `Δ loss dinámico`.
4. Gráfico de barras comparando pérdida total sin shock vs con shock para ambos modelos.
5. Mensaje de alerta si el modelo dinámico absorbe mejor el shock (pérdida relativa menor).
- Precondición: `assert shock_debt >= 0 and shock_income >= 0`.
- Precondición: `assert isinstance(df, pd.DataFrame) and not df.empty`.

---

### 5.5 — Actualización de `settings.py`

Añadir las rutas nuevas al objeto `Settings`:

```python
EVAL_TRANSACTIONS_PATH: Path = DB_PATH / "eval_transactions.csv"
EVAL_FEATURES_PATH: Path = DB_PATH / "eval_features.csv"
EVAL_SEED: int = 43
```

---

## Tests pytest

**Archivos:** `tests/conftest.py` + archivos por módulo

### `conftest.py`

- Fixtures: `df_raw`, `df_features`, `df_eval`, `A_mat`, `B_mat`, `C_mat`, `K_mat`, `kalman_instance`, `modelo_logistico`.
- Fixture `db_tmp`: SQLite temporal en `tmp_path`.
- Fixture `customer_builder`: instancia de `CustomerBuilder` configurada.
- Fixture `customer_director`: instancia de `CustomerDirector`.
- `SEED = 42` en fixtures de datos de entrenamiento; `SEED = 43` en fixtures de evaluación.

### `test_builder.py`

- Verifica que `CustomerBuilder` lanza `AssertionError` si `build()` se llama sin configurar atributos.
- Verifica que `build()` retorna exactamente 24 dicts.
- Verifica que cada dict contiene las columnas requeridas (`customer_id`, `month`, `income`, etc.).
- Verifica que `CustomerDirector.construir_dataset(42)` produce 12 000 filas.
- Verifica distribución de arquetipos: `good ≈ 200`, `recurrent ≈ 125`, etc.
- Verifica que con seed distinto el resultado es diferente.

### `test_features.py`

- Verifica que `ratio_deuda_ingreso_ma` usa ventana de 3 meses correcta.
- Verifica que `tendencia_utilizacion` es 0.0 cuando hay 1 solo punto.
- Verifica que `income` NaN se imputa antes de calcular ratio.
- Verifica que `customer_id` y `month` no cambian tras el pipeline.
- Verifica que `db/features_dinamicos.csv` existe con columnas esperadas.
- Verifica que `generar_features` produce el mismo resultado para `eval_transactions.csv`.

### `test_kalman.py`

- Verifica que `paso()` produce `x_hat` con forma `(3, 1)`.
- Verifica que con `y_t` con NaN el estado se propaga sin actualización.
- Verifica que `P` permanece semidefinida positiva tras varios pasos.
- Si `P` tiene autovalores negativos → el test **FALLA** con mensaje descriptivo.
- Verifica convergencia: tras 20 pasos con sistema estable, `trace(P)` decrece.

### `test_controlador.py`

- Verifica que `K` tiene forma `(1, 3)`.
- Verifica que `limit_recomendado` está en `[0, credit_limit_max]`.
- Verifica que `score_dinamico` ∈ `[0, 1]`.
- Verifica que con Q_lqr de alta penalización en deuda el límite recomendado sea conservador.

### `test_logistico.py`

- Verifica que `predict_proba` retorna float en `[0, 1]`.
- Verifica que el modelo no se reentrena si el `.pkl` ya existe.
- Verifica que métricas `auc`, `gini`, `ks` están en rangos válidos.
- Verifica que el pipeline incluye `scaler` y `clf`.

### `test_integracion.py`

- Marca: `@pytest.mark.slow`.
- Ejecuta pipeline completo: generación (via `CustomerDirector`) → features → matrices → Kalman → LQR → backtesting.
- Verifica que `db/comparacion_modelos.json` existe al final.
- Verifica que tabla `decisions` tiene exactamente 12 000 filas.
- Verifica AUC logístico > 0.6.
- Verifica que `eval_features.csv` existe y tiene misma estructura que `features_dinamicos.csv`.
- Verifica que el logger no lanza excepciones durante la ejecución completa.

---

## Mapa de dependencias entre artefactos

```
modelos/dominio/customer_builder.py + customer_director.py
    → (usado por) scripts/transacciones.py

scripts/transacciones.py
    → db/raw_transactions.csv   (SEED=42)
    → db/eval_transactions.csv  (SEED=43)

modelos/features/pipeline.py  ← db/raw_transactions.csv
    → db/features_dinamicos.csv

modelos/features/pipeline.py  ← db/eval_transactions.csv
    → db/eval_features.csv

modelos/estatico/logistico.py  ← db/features_dinamicos.csv
    → db/modelo_logistico.pkl
    → db/metricas_baseline.json

modelos/dinamico/identificacion.py  ← db/features_dinamicos.csv
    → db/matrices_sistema.npz
    → db/matrices_sistema.json   (scale params)

modelos/dinamico/kalman.py  ← matrices_sistema.npz
    (instanciado en runtime; sin artefacto de salida propio)

modelos/dinamico/controlador.py  ← matrices_sistema.npz
    (instanciado en runtime; sin artefacto de salida propio)

gui/infraestructura/db.py
    → db/credito.db  (esquema)

modelos/evaluacion/backtesting.py
    ← db/features_dinamicos.csv
    ← db/modelo_logistico.pkl
    ← db/matrices_sistema.npz
    ← db/credito.db
    → db/comparacion_modelos.json
    → db/credito.db  (pobla tablas)

utils/logger.py
    ← (importado por todos los módulos)
    (sin artefacto de salida; escribe logs a consola / archivo)

gui/app.py
    ← db/modelo_logistico.pkl
    ← db/matrices_sistema.npz + matrices_sistema.json
    ← db/eval_features.csv   → st.session_state['eval_df']
    ← db/comparacion_modelos.json

gui/paginas/*.py
    ← st.session_state['eval_df']
    ← modelos cargados en memoria via app.py
    (sin artefactos de salida; render en Streamlit)

gui/componentes/shock_simulator.py
    ← st.session_state['eval_df']  (copia modificada en memoria)
    (sin artefactos de salida)
```

---

## Criterios de aceptación globales

Un agente puede marcar el proyecto como **completado** cuando se cumplan **todos** los siguientes criterios:

| # | Criterio | Verificación |
|---|---|---|
| 1 | `CustomerBuilder.build()` genera 24 filas correctas por cliente | `pytest tests/test_builder.py` |
| 2 | `CustomerDirector.construir_dataset(42)` genera 12 000 filas con distribución de arquetipos correcta | `pytest tests/test_builder.py` |
| 3 | `db/raw_transactions.csv` tiene 12 000 filas (SEED=42) | `wc -l db/raw_transactions.csv` = 12 001 |
| 4 | `db/eval_transactions.csv` tiene 12 000 filas (SEED=43) | `wc -l db/eval_transactions.csv` = 12 001 |
| 5 | `db/features_dinamicos.csv` y `db/eval_features.csv` contienen las 3 columnas de features nuevas | `head -1 db/*_features*.csv` |
| 6 | `db/modelo_logistico.pkl` existe y AUC > 0.6 | `db/metricas_baseline.json` |
| 7 | `db/matrices_sistema.npz` tiene claves A (3×3), B (3×1), C (2×3) | `np.load('db/matrices_sistema.npz').files` + shape check |
| 8 | Filtro de Kalman converge sin autovalores negativos en P | `pytest tests/test_kalman.py` |
| 9 | Score dinámico ∈ [0, 1] para todos los clientes | `pytest tests/test_controlador.py` |
| 10 | `db/comparacion_modelos.json` contiene Gini, KS, AUC y pérdida de ambos modelos | Inspección JSON |
| 11 | `modelos/evaluacion/` (sin typo) existe y todos los imports funcionan | `python -c "from modelos.evaluacion.backtesting import run_backtesting"` |
| 12 | `utils/logger.py` configurable una vez; todos los módulos lo usan | `grep -r "get_logger" modelos/ gui/ utils/` |
| 13 | `streamlit run main.py` lanza sin errores y carga las 4 pestañas | Inspección visual |
| 14 | Pestaña "Análisis Descriptivo" muestra resumen estadístico, histogramas y alertas del eval_df | Inspección visual |
| 15 | Pestaña "Modelo Estático" muestra AUC-ROC, KS, Gini, PSI y scatter de eval_df | Inspección visual |
| 16 | Pestaña "Modelo Dinámico" muestra estados Kalman estimados vs reales para cliente seleccionado | Inspección visual |
| 17 | Pestaña "Comparación" muestra tabla de métricas side-by-side y simulación de shock funcional | Inspección visual |
| 18 | Shock simulator recalcula pérdidas de ambos modelos al activarse | Inspección visual + prueba interactiva |
| 19 | `pytest tests/ -m "not slow"` pasa sin errores | Salida pytest |
| 20 | `pytest tests/ --cov=modelos --cov=utils` alcanza cobertura ≥ 80% | `pytest --cov` |
| 21 | Ninguna función supera 60 líneas | Análisis estático |
| 22 | Toda variable tiene tipo explícito | Revisión de código / mypy |
| 23 | Mínimo 2 aserciones por función | Revisión de código |
| 24 | `docs/tareas.md` refleja estado actualizado de todas las tareas | Inspección |