# Tareas — dashboard-dinamic-score-system

> Última actualización: 2026-06-27

## Leyenda de estado
- `⬜ pendiente` — no iniciada
- `🔄 en curso` — en desarrollo
- `✅ completado` — terminada y verificada
- `❌ bloqueada` — bloqueada por dependencia no resuelta

---

## Fase 0 — Bootstrapping

| ID | Tarea | Estado | Dependencias | Archivo(s) |
|---|---|---|---|---|
| T0.1 | Crear estructura de directorios y `__init__.py` vacíos | ✅ completado | — | Todos los directorios |
| T0.2 | Crear `requirements.txt` con dependencias fijadas | ✅ completado | — | `requirements.txt` |
| T0.3 | Crear `docs/tareas.md` y `docs/plan-implementacion.md` | ✅ completado | — | `docs/` |
| T0.4 | Validar Python 3.12, instalar dependencias, verificar imports | ✅ completado | T0.2 | — |

---

## Fase 1.1 — Generación de datos sintéticos

| ID | Tarea | Estado | Dependencias | Archivo(s) |
|---|---|---|---|---|
| T1.1.1 | Añadir flag `--force` y validaciones al script existente | 🔄 en curso | T0.4 | `scripts/transacciones.py` |
| T1.1.2 | Añadir aserciones (N_CUSTOMERS, shape del CSV) | ⬜ pendiente | T1.1.1 | `scripts/transacciones.py` |
| T1.1.3 | Ejecutar y verificar CSV (12 000 filas, distribución arquetipos) | ⬜ pendiente | T1.1.2 | `db/raw_transactions.csv` |

---

## Fase 1.2 — Feature Engineering

| ID | Tarea | Estado | Dependencias | Archivo(s) |
|---|---|---|---|---|
| T1.2.1 | Implementar `imputar_income` | ⬜ pendiente | T1.1.3 | `modelos/features/pipeline.py` |
| T1.2.2 | Implementar `calcular_ratio_deuda_ingreso_ma` | ⬜ pendiente | T1.2.1 | `modelos/features/pipeline.py` |
| T1.2.3 | Implementar `calcular_tendencia_utilizacion` | ⬜ pendiente | T1.2.1 | `modelos/features/pipeline.py` |
| T1.2.4 | Implementar `calcular_volatilidad_pagos` | ⬜ pendiente | T1.2.1 | `modelos/features/pipeline.py` |
| T1.2.5 | Implementar `generar_features` (orquestador) y verificar CSV de salida | ⬜ pendiente | T1.2.2, T1.2.3, T1.2.4 | `modelos/features/pipeline.py`, `db/features_dinamicos.csv` |

---

## Fase 1.3 — Modelo Logístico Baseline

| ID | Tarea | Estado | Dependencias | Archivo(s) |
|---|---|---|---|---|
| T1.3.1 | Implementar `_cargar_features` y `_construir_pipeline` | ⬜ pendiente | T1.2.5 | `modelos/estatico/logistico.py` |
| T1.3.2 | Implementar `_calcular_metricas` (AUC, Gini, KS) | ⬜ pendiente | T1.3.1 | `modelos/estatico/logistico.py` |
| T1.3.3 | Implementar `entrenar_y_guardar` con split estratificado 80/20 | ⬜ pendiente | T1.3.2 | `modelos/estatico/logistico.py` |
| T1.3.4 | Implementar `cargar_modelo` y `predict_proba` | ⬜ pendiente | T1.3.3 | `modelos/estatico/logistico.py` |
| T1.3.5 | Implementar `inicializar` (singleton .pkl) | ⬜ pendiente | T1.3.4 | `modelos/estatico/logistico.py` |
| T1.3.6 | Verificar `.pkl` y `metricas_baseline.json` generados correctamente | ⬜ pendiente | T1.3.5 | `db/modelo_logistico.pkl`, `db/metricas_baseline.json` |

---

## Fase 2.1 — Identificación del sistema

| ID | Tarea | Estado | Dependencias | Archivo(s) |
|---|---|---|---|---|
| T2.1.1 | Implementar `normalizar_datos` | ⬜ pendiente | T1.2.5 | `modelos/dinamico/identificacion.py` |
| T2.1.2 | Implementar `construir_matrices_regresion` | ⬜ pendiente | T2.1.1 | `modelos/dinamico/identificacion.py` |
| T2.1.3 | Implementar `identificar_AB` (regresión mínimos cuadrados) | ⬜ pendiente | T2.1.2 | `modelos/dinamico/identificacion.py` |
| T2.1.4 | Implementar `identificar_C` | ⬜ pendiente | T2.1.3 | `modelos/dinamico/identificacion.py` |
| T2.1.5 | Implementar `verificar_mse` (< 0.05 en escala normalizada) | ⬜ pendiente | T2.1.4 | `modelos/dinamico/identificacion.py` |
| T2.1.6 | Implementar `guardar_matrices` y `cargar_matrices` | ⬜ pendiente | T2.1.5 | `modelos/dinamico/identificacion.py` |
| T2.1.7 | Implementar `identificar` (orquestador con cache) y verificar `matrices_sistema.npz` | ⬜ pendiente | T2.1.6 | `db/matrices_sistema.npz` |

---

## Fase 2.2 — Filtro de Kalman

| ID | Tarea | Estado | Dependencias | Archivo(s) |
|---|---|---|---|---|
| T2.2.1 | Implementar clase `FiltroKalman.__init__` | ⬜ pendiente | T2.1.7 | `modelos/dinamico/kalman.py` |
| T2.2.2 | Implementar `predecir` (predicción a priori) | ⬜ pendiente | T2.2.1 | `modelos/dinamico/kalman.py` |
| T2.2.3 | Implementar `actualizar` con manejo de NaN (EARS-K03) | ⬜ pendiente | T2.2.2 | `modelos/dinamico/kalman.py` |
| T2.2.4 | Implementar `_simetrizar_covarianza` con detección de autovalores negativos | ⬜ pendiente | T2.2.3 | `modelos/dinamico/kalman.py` |
| T2.2.5 | Implementar `paso` y `ejecutar_secuencia` | ⬜ pendiente | T2.2.4 | `modelos/dinamico/kalman.py` |

---

## Fase 2.3 — Controlador LQR

| ID | Tarea | Estado | Dependencias | Archivo(s) |
|---|---|---|---|---|
| T2.3.1 | Implementar `matrices_costo_default` | ⬜ pendiente | T2.2.5 | `modelos/dinamico/controlador.py` |
| T2.3.2 | Implementar `calcular_ganancia_lqr` con DARE de scipy | ⬜ pendiente | T2.3.1 | `modelos/dinamico/controlador.py` |
| T2.3.3 | Implementar `decidir_limite` con saturación en [0, max] | ⬜ pendiente | T2.3.2 | `modelos/dinamico/controlador.py` |
| T2.3.4 | Implementar `score_dinamico` normalizado [0,1] | ⬜ pendiente | T2.3.3 | `modelos/dinamico/controlador.py` |

---

## Infraestructura SQLite

| ID | Tarea | Estado | Dependencias | Archivo(s) |
|---|---|---|---|---|
| T3.0.1 | Implementar `inicializar_db` y `obtener_conexion` | ⬜ pendiente | T0.1 | `gui/infraestructura/db.py` |
| T3.0.2 | Implementar `CustomerRepository` | ⬜ pendiente | T3.0.1 | `gui/infraestructura/repositorios/cliente_repo.py` |
| T3.0.3 | Implementar `EstadoRepository` | ⬜ pendiente | T3.0.1 | `gui/infraestructura/repositorios/estado_repo.py` |
| T3.0.4 | Implementar `DecisionRepository` | ⬜ pendiente | T3.0.1 | `gui/infraestructura/repositorios/decision_repo.py` |

---

## Fase 3 — Backtesting

| ID | Tarea | Estado | Dependencias | Archivo(s) |
|---|---|---|---|---|
| T3.1 | Implementar `calcular_perdida_mes` | ⬜ pendiente | T1.3.6, T2.3.4, T3.0.4 | `modelos/evaluacion/backtesting.py` |
| T3.2 | Implementar `simular_modelo_logistico` | ⬜ pendiente | T3.1 | `modelos/evaluacion/backtesting.py` |
| T3.3 | Implementar `simular_modelo_dinamico` | ⬜ pendiente | T3.1 | `modelos/evaluacion/backtesting.py` |
| T3.4 | Implementar `calcular_gini_ks_auc` y `calcular_psi` | ⬜ pendiente | T3.2, T3.3 | `modelos/evaluacion/backtesting.py` |
| T3.5 | Implementar `calcular_psi_serie` | ⬜ pendiente | T3.4 | `modelos/evaluacion/backtesting.py` |
| T3.6 | Implementar `comparar_modelos` con advertencia si reducción < 5% | ⬜ pendiente | T3.5 | `modelos/evaluacion/backtesting.py` |
| T3.7 | Implementar `guardar_comparacion` y `poblar_sqlite` | ⬜ pendiente | T3.6 | `modelos/evaluacion/backtesting.py` |
| T3.8 | Implementar `ejecutar_backtesting` (orquestador) y verificar salidas | ⬜ pendiente | T3.7 | `db/comparacion_modelos.json`, `db/credito.db` |

---

## Fase 4 — Interfaz gráfica

### CSS y base

| ID | Tarea | Estado | Dependencias | Archivo(s) |
|---|---|---|---|---|
| T4.0.1 | Crear `gui/global.css` con reset, base 10px, variables comunes | ⬜ pendiente | T0.1 | `gui/global.css` |
| T4.0.2 | Crear `gui/cliente/styles.css` con variables de color del cliente | ⬜ pendiente | T4.0.1 | `gui/cliente/styles.css` |
| T4.0.3 | Crear `gui/banco/styles.css` con variables de color del banco | ⬜ pendiente | T4.0.1 | `gui/banco/styles.css` |
| T4.0.4 | Crear `main.py` con arranque NiceGUI, inicialización DB, registro de rutas | ⬜ pendiente | T3.8, T4.0.1 | `main.py` |

### Dashboard Cliente

| ID | Tarea | Estado | Dependencias | Archivo(s) |
|---|---|---|---|---|
| T4.1.1 | Implementar `SelectorCliente` (dropdown con búsqueda) | ⬜ pendiente | T4.0.2, T3.0.2 | `gui/cliente/componentes/selector_cliente.py` |
| T4.1.2 | Implementar `ScoreGauge` (gauge 0-100, alerta si <40) | ⬜ pendiente | T4.1.1 | `gui/cliente/componentes/score_gauge.py` |
| T4.1.3 | Implementar `TrayectoriaChart` (4 subgráficos sincronizados, marcadores default) | ⬜ pendiente | T4.1.1 | `gui/cliente/componentes/trayectoria_chart.py` |
| T4.1.4 | Implementar `TablaTransacciones` (últimas 6 transacciones) | ⬜ pendiente | T4.1.1 | `gui/cliente/componentes/tabla_transacciones.py` |
| T4.1.5 | Implementar `dashboard_cliente.py` (layout grid + callbacks) | ⬜ pendiente | T4.1.1, T4.1.2, T4.1.3, T4.1.4 | `gui/cliente/paginas/dashboard_cliente.py` |

### Dashboard Banco

| ID | Tarea | Estado | Dependencias | Archivo(s) |
|---|---|---|---|---|
| T4.2.1 | Implementar `FiltroArquetipo` | ⬜ pendiente | T4.0.3, T3.0.2 | `gui/banco/componentes/filtro_arquetipo.py` |
| T4.2.2 | Implementar `FeaturesAgregados` (3 métricas + histograma) | ⬜ pendiente | T4.2.1 | `gui/banco/componentes/features_agregados.py` |
| T4.2.3 | Implementar `PanelMetricas` (tabla side-by-side logístico vs dinámico) | ⬜ pendiente | T4.2.1 | `gui/banco/componentes/panel_metricas.py` |
| T4.2.4 | Implementar `AutovaloresPlot` (plano complejo + barras K) | ⬜ pendiente | T4.2.1 | `gui/banco/componentes/autovalores_plot.py` |
| T4.2.5 | Implementar `PSIChart` (línea temporal con umbral 0.25 y alertas) | ⬜ pendiente | T4.2.1 | `gui/banco/componentes/psi_chart.py` |
| T4.2.6 | Implementar `SimulacionContrafactual` (con vs sin LQR) | ⬜ pendiente | T4.2.1 | `gui/banco/componentes/simulacion_contrafactual.py` |
| T4.2.7 | Implementar `dashboard_banco.py` (layout 4 secciones + callbacks) | ⬜ pendiente | T4.2.1-T4.2.6 | `gui/banco/paginas/dashboard_banco.py` |

---

## Tests pytest

| ID | Tarea | Estado | Dependencias | Archivo(s) |
|---|---|---|---|---|
| T5.1 | Crear `conftest.py` con fixtures reutilizables | ⬜ pendiente | T1.2.5, T2.2.5 | `tests/conftest.py` |
| T5.2 | Implementar `test_features.py` | ⬜ pendiente | T5.1 | `tests/test_features.py` |
| T5.3 | Implementar `test_kalman.py` (incl. test autovalores negativos) | ⬜ pendiente | T5.1 | `tests/test_kalman.py` |
| T5.4 | Implementar `test_controlador.py` | ⬜ pendiente | T5.1 | `tests/test_controlador.py` |
| T5.5 | Implementar `test_logistico.py` | ⬜ pendiente | T5.1 | `tests/test_logistico.py` |
| T5.6 | Implementar `test_integracion.py` (pipeline completo 500 clientes) | ⬜ pendiente | T5.1, T3.8 | `tests/test_integracion.py` |
| T5.7 | Ejecutar `pytest tests/ --cov=modelos` y verificar cobertura ≥ 80% | ⬜ pendiente | T5.2-T5.6 | — |

---

## Resumen de progreso

| Fase | Total tareas | Completadas | Pendientes | Bloqueadas |
|---|---|---|---|---|
| Fase 0 — Setup | 4 | 1 | 3 | 0 |
| Fase 1.1 — Datos | 3 | 0 | 3 | 0 |
| Fase 1.2 — Features | 5 | 0 | 5 | 0 |
| Fase 1.3 — Logístico | 6 | 0 | 6 | 0 |
| Fase 2.1 — Identificación | 7 | 0 | 7 | 0 |
| Fase 2.2 — Kalman | 5 | 0 | 5 | 0 |
| Fase 2.3 — LQR | 4 | 0 | 4 | 0 |
| Infraestructura SQLite | 4 | 0 | 4 | 0 |
| Fase 3 — Backtesting | 8 | 0 | 8 | 0 |
| Fase 4 — GUI | 16 | 0 | 16 | 0 |
| Tests | 7 | 0 | 7 | 0 |
| **TOTAL** | **69** | **1** | **68** | **0** |