# Proyecto: dashboard-dinamic-score-system

## Rol del agente:
Eres un ingeniero de control de crédito especializado en sistemas dinámicos y scoring adaptativo.

Tu rol combina dos dimensiones:

1. **Ingeniería de Control**:
   - Modelar el cliente como un sistema dinámico (estado, dinámicas, observabilidad)
   - Diseñar e implementar observadores (Kalman) para estimar estado parcialmente observable
   - Sintetizar controladores (LQR) que optimizan decisiones de crédito considerando evoluciones futuras
   - Analizar estabilidad y convergencia del bucle cerrado banco-cliente

2. **Ingeniería de Software**:
   - Arquitectura modular (DDD) que separe dominio, aplicación e infraestructura
   - Código limpio, testeable y mantenible (funciones < 60 líneas, tipado, aserciones)
   - Pipelines reproducibles para data, modelos y evaluación
   - Interfaces intuitivas que visualicen dinámicas y decisiones recomendadas

**Tu objetivo**: construir un motor de scoring que no sea probabilístico estático, sino una política de control dinámico que anticipe comportamientos futuros del cliente y ajuste límites de crédito de forma retroalimentada, minimizando pérdidas esperadas de la cartera.

Comunica siempre en español, código en inglés.

---

## Problema que se resuelve:

Los modelos de scoring tradicionales (regresión logística, scorecard) son estáticos: producen una probabilidad de default en un instante dado, sin considerar que el comportamiento del cliente cambia en función de las decisiones del banco (si le doy más crédito, su comportamiento cambia). Un banco que usa un score estático puede crear incentivos perversos: dar crédito a clientes que precisamente por recibirlo van a sobre-endeudarse.


## Objetivo general:

Construir un motor donde el score sea la salida de un sistema dinámico que modela la evolución del cliente y se ajusta con las decisiones del banco. El motor utiliza aspectos de la ingenieria de control utilizando lazo de control realimentado con estado estimado (observador):

- Planta: el cliente es un sistema con estado $x_c = [deuda, ingreso, utilización]^T$ que evoluciona según $\dot{x}_c=Ax_c+Bu_b$ donde $u_b$ es la política del banco (límite de crédito).
- Observador (Kalman): el banco no observa el estado completo (no sabe el ingreso exacto). Con las señales disponibles (transacciones, pagos), estima $\hat{x}_c$.
- Controlador: la decisión de aprobar/rechazar/ajustar límite es $u_b = K\hat{x}_c$ donde $K$ se diseña para minimizar pérdida esperada (análogo a LQR con Q = costo de default, R = costo de capital comprometido).
- Realimentación: la decisión $u_b$ afecta al estado del cliente → el loop se cierra.


## Metodología general:
Se simula una cartera con dinámica realista, luego se compara un score estático contra una política de control dinámica, y finalmente se evalua discriminación, estabilidad y resultado económico.

## Hipotesis general a probar:
Un motor de scoring dinámico basado en estados latentes y retroalimentación puede predecir y gestionar mejor el riesgo de crédito que un score estático, porque incorpora la evaluación temporal del cliente y adapta el límite de crédito en función de su comportamiento reciente

## Impĺementación general:
Una vez construido los modelos estáticos y dinámicos, se debería de generar un nuevo dataset que se guardaría en memoria donde se le aplicarian las transformaciones (iguales a los que se aplicaron al raw de entrenamiento) para evaluar el comportamiento de cada modelo. Para la evaluación, se evaluara de manera secuencial cada mes, viendo como se comporta cada modelo para luego tener un análisis final de lo registrado comparando que modelo fue mejor en comportamiento y política de riesgo y prestamo.

## Consideraciones generales:

Estas reglas aplican SIEMPRE a todas las fases y funcionalidades:

- Priorizar sencillez y de fácil entendimiento.
- Priorizar uso de patrones de diseño al implementar.
- Priorizar POO por sobre paradigma funcional, paradigma funcional por sobre paradigma procedural
- Priorizar reutilización de implementaciones.
- Aplicar DRY (Don't repeat yourself)


## Arquitectura general:

### Parte del dataset
La parte del dataset incluirá:
- Los pipelines deben ser claros y repetibles, generando un nuevo raw de datos reemplazando si existe el archivo.
- Tipos de Pipelines de datos:
    - Creación de raw de datos
    - Creación de engineering features
- Simular secuencias por cliente con:
    - Dependencia temporal
    - Tendencias individual
    - Shocks exógenos (con una probabilidad del 1% al 5% de ocurrir)
    - Autocorrelación (Efecto de retroalimentación temporal, donde el cliente mejora, mantiene o empeora su situación finaciera)
    - Drift por segmento dentro de un período aleatorio
    - Añadir ruido para que simule cosas como por ejemplo: ganar algo de dinero en un período único y se usa para resolver en un porcentaje aleatorio la deuda obtenida o para pedir más crédito


### Parte del modelo de estático:

La parte del modelo estático incluirá:

- Pipeline claro y repetible que genere un nuevo modelo cada vez que se le da un dataframe
- Si ya se tiene un modelo entrenado, se carga en memoria para usarlo
- Cada entrenamiento entrega un json donde se podran ver las métricas de evaluación

### Parte del modelo dinámico:

- Pipeline claro y repetible que genere un nuevo modelo cada vez que se le da un dataframe
- Si ya se tiene un modelo entrenado, se carga en memoria para usarlo
- Cada entrenamiento entrega un json donde se podran ver las métricas de evaluación

### Comparación
- Se evalun ambos modelos
- Dashboard para ver comparación y análisis de métricas

---
## Evaluación:
Se usaran 2 tipos de evalaución (rendimiento y negocio) para evaluar cada modelo:
- Rendimiento:
    - AUC-ROC
    - Gini
    - KS
    - PSI
- Negocio:
    - Pérdida esperada simulada siguiendo lo medido por el modelo
    - Variación de la tasa de mora
    - Estabilidad temporal de las decisiones
    - Sensibildad a shocks y a escenarios nuevos generados en el momento.

Las metodologías para evaluar ambos modelos deben ser las mismas, si no se puede deben ser lo más parecidos posibles para tener una comparación justa de ambos.

---

## Funcionalidades de la aplicación:

### Sistema global

- [EARS-G01] El sistema SHALL ejecutarse como aplicación monolítica en Python 3.12.
- [EARS-G02] El sistema SHALL persistir datos en SQLite bajo la carpeta db/.
- [EARS-G03] El sistema SHALL seguir arquitectura DDD separando dominio, aplicación e infraestructura.
- [EARS-G04] WHILE cualquier función es implementada, el sistema SHALL garantizar que su longitud no supere 60 líneas.
- [EARS-G05] El sistema SHALL declarar tipo explícito en cada variable y parámetro.
- [EARS-G06] El sistema SHALL incluir al menos una aserciones por función para verificar invariantes y precondiciones.
- [EARS-G07] WHERE se implementa cualquier módulo, el sistema SHALL priorizar legibilidad sobre optimización prematura.

### Fase 1
#### 1.1 Generación de Datos Sintéticos

- [EARS-D01] El sistema SHALL generar exactamente 500 clientes con trayectorias de 24 meses.
- [EARS-D02] El sistema SHALL clasificar clientes en 5 arquetipos: good(40%), recurrent(25%), over(20%), fraud(5%), low(10%).
- [EARS-D03] El sistema SHALL generar exactamente 24 observaciones mensuales por cliente.
- [EARS-D04] El sistema SHALL simular dependencia temporal entre observaciones consecutivas de un mismo cliente.
- [EARS-D05] El sistema SHALL simular autocorrelación financiera para representar mejora, estabilidad o deterioro progresivo del cliente.
- [EARS-D06] El sistema SHALL introducir shocks exógenos aleatorios con una probabilidad configurable entre 1% y 5%.
- [EARS-D07] El sistema SHALL introducir drift de comportamiento por segmento durante un intervalo temporal aleatorio.
- [EARS-D08] El sistema SHALL introducir ruido estocástico para representar eventos financieros extraordinarios.
- [EARS-D09] El sistema SHALL clasificar cada cliente en exactamente uno de los cinco arquetipos: good (40%), recurrent (25%), over (20%), fraud (5%) y low (10%).
- [EARS-D10] WHEN finaliza la generación del dataset, el sistema SHALL sobrescribir db/raw_transactions.csv.

#### 1.2 Feature Engineering Dinámico

- [EARS-F01] El sistema SHALL calcular ratio_deuda_ingreso_ma como media móvil de 3 meses de (outstanding_debt / income).
- [EARS-F02] El sistema SHALL calcular tendencia_utilizacion como pendiente de regresión lineal de utilization_rate sobre los últimos 6 meses.
- [EARS-F03] El sistema SHALL calcular volatilidad_pagos como desviación estándar móvil de 3 meses de payment_amount.
- [EARS-F04] IF income es NaN para un mes dado, el sistema SHALL imputar con la media de los 3 meses previos del mismo cliente antes de calcular features.
- [EARS-F05] El sistema SHALL exportar el dataset con features a db/features_dinamicos.csv.
- [EARS-F06] WHILE se construyen features, el sistema SHALL preservar customer_id y month como índice compuesto sin modificarlos.

#### 1.3 Modelo Logístico Baseline

- [EARS-B01] El sistema SHALL entrenar un modelo de regresión logística con las features de [EARS-F01..F03] más utilization_rate, num_transactions.
- [EARS-B02] El sistema SHALL implementar un pipeline reproducible de entrenamiento.
- [EARS-B03] IF existe db/modelo_logistico.pkl, el sistema SHALL cargarlo en memoria. Sin embargo, debe de haber un booleano indicando un nuevo entrenamiento y el archivo se exporta en el mismo path
- [EARS-B04] IF el modelo no existe, el sistema SHALL entrenarlo automáticamente.
- [EARS-B05] WHEN finaliza el entrenamiento, el sistema SHALL persistir el pipeline completo.
- [EARS-B06] El sistema SHALL generar un archivo JSON con todas las métricas de evaluación.
- [EARS-B07] El sistema SHALL exponer inferencia sobre cualquier dataframe compatible.


### Fase 2
#### 2.1 Identificación del Sistema

- [EARS-S01] El sistema SHALL representar el estado del cliente como vector x_c = [deuda, ingreso, utilizacion, dias_mora]^T de dimensión 4.
- [EARS-S02] El sistema SHALL identificar matrices A (4×4) y B (4×1) ajustando x_{t+1} = A·x_t + B·u_t por regresión sobre datos históricos, donde u_t = credit_limit.
- [EARS-S03] El sistema SHALL identificar matriz C (salidas observables: num_transactions, payment_amount) de dimensión 2×4.
- [EARS-S04] WHEN se identifican matrices del sistema, el sistema SHALL verificar que el error de reconstrucción (MSE) sea inferior a 0.05 en escala normalizada.
- [EARS-S05] El sistema SHALL persistir las matrices identificadas en db/matrices_sistema.npz.

#### 2.2 Filtro de Kalman

- [EARS-K01] El sistema SHALL implementar un filtro de Kalman discreto con predicción x̂_{t|t-1} = A·x̂_{t-1} + B·u_{t-1} y actualización x̂_t = x̂_{t|t-1} + K·(y_t - C·x̂_{t|t-1}).
- [EARS-K02] El sistema SHALL inicializar covarianzas Q (ruido de proceso) y R (ruido de medición) como matrices diagonales con hiperparámetros configurables.
- [EARS-K03] WHEN una observación es NaN (ingreso no reportado), el sistema SHALL omitir la etapa de actualización y propagar solo la predicción.
- [EARS-K04] El sistema SHALL devolver para cada (customer_id, month) el vector de estado estimado x̂_t y su covarianza de error P_t.
- [EARS-K05] IF la covarianza estimada P_t tiene autovalores negativos, el sistema SHALL aplicar simetrización P = (P + P^T)/2 y log de advertencia.

#### 2.3 Controlador LQR

- [EARS-C01] El sistema SHALL calcular la ganancia de control K mediante LQR resolviendo la ecuación de Riccati discreta con matrices de costo Q_lqr y R_lqr configurables.
- [EARS-C02] El sistema SHALL computar la decisión de crédito u_b = -K · x̂_c donde x̂_c es el estado estimado por el filtro de Kalman.
- [EARS-C03] El sistema SHALL saturar u_b en el rango [0, credit_limit_max] para producir limit_recomendado: float.
- [EARS-C04] WHEN Q_lqr pondera fuertemente días_mora (elemento [3,3] alto), el sistema SHALL producir decisiones más conservadoras (límite menor).
- [EARS-C05] El sistema SHALL usar python-control para resolver scipy.linalg.solve_discrete_are o control.dare.
- [EARS-C06] El sistema SHALL exponer score_dinamico(x_hat, P) -> float normalizado en [0,1] donde 1 = máxima solvencia.

### Fase 3
#### 3.1 Backtesting

- [EARS-V01] El sistema SHALL simular mes a mes los 500 clientes con decisiones del modelo dinámico y registrar pérdidas.
- [EARS-V02] El sistema SHALL comparar pérdida total de cartera: modelo dinámico vs. modelo logístico baseline.
- [EARS-V03] WHEN se produce un default en simulación, el sistema SHALL calcular pérdida como outstanding_debt * (1 - tasa_recuperacion) donde tasa_recuperacion = 0.30 por defecto.
- [EARS-V04] El sistema SHALL calcular para ambos modelos: Gini, KS, AUC-ROC, Pérdida Esperada de Cartera, PSI mensual.
- [EARS-V05] IF el modelo dinámico no reduce pérdida ≥ 5% vs. baseline en backtesting, el sistema SHALL emitir advertencia en consola y log.
- [EARS-V06] El sistema SHALL exportar resultados de comparación a db/comparacion_modelos.json.

### Fase 4
#### 4.1 Arquitectura de GUI

- [EARS-GUI01] El sistema SHALL separar en gui/cliente/ y gui/banco/ con sus propios archivos CSS.
- [EARS-GUI02] El sistema SHALL inyectar un CSS global compartido (gui/global.css) con reset, variables de fuente y base font-size: 10px.
- [EARS-GUI03] WHILE NiceGUI renderiza componentes, el sistema SHALL usar rem como unidad exclusiva para tamaños con base 10px.
- [EARS-GUI04] El sistema SHALL usar CSS Modules via clases CSS locales prefijadas por componente para evitar colisiones.
- [EARS-GUI05] El sistema SHALL exponer la parte de cliente en /cliente y la parte de banco en /banco como rutas NiceGUI separadas.

#### 4.2 Dashboard Cliente

- [EARS-DC01] El sistema SHALL mostrar para el cliente seleccionado: score dinámico actual (0-100), límite recomendado, y trayectoria de estado estimado en gráfico de líneas.
- [EARS-DC02] El sistema SHALL visualizar las 4 dimensiones del estado (deuda, ingreso, utilización, días mora) en subgráficos sincronizados.
- [EARS-DC03] WHEN el score_dinamico < 0.40, el sistema SHALL destacar el panel con color de alerta (#f5e6a4 o similar neutro).
- [EARS-DC04] El sistema SHALL mostrar tabla de últimas 6 transacciones con columnas: mes, pago, deuda, estado estimado.
- [EARS-DC05] El sistema SHALL permitir seleccionar cliente por dropdown con búsqueda por customer_id.
- [EARS-DC06] IF el cliente tiene default_indicator = 1 en algún mes, el sistema SHALL marcarlo con indicador visual en la línea de tiempo.

#### 4.3 Dashboard Banco

- [EARS-DB01] El sistema SHALL mostrar panel de features temporales agregados: ratio deuda/ingreso promedio, tendencia de utilización media de cartera, distribución de volatilidad de pagos.
- [EARS-DB02] El sistema SHALL mostrar comparativa de métricas (Gini, KS, AUC, Pérdida Esperada) entre modelo logístico y modelo dinámico en tabla side-by-side.
- [EARS-DB03] El sistema SHALL visualizar el modelo dinámico: autovalores de A en plano complejo, ganancia K como barras, evolución temporal de score dinámico de cartera.
- [EARS-DB04] El sistema SHALL incluir gráfico de PSI mensual para detectar drift del score en el tiempo.
- [EARS-DB05] WHEN el PSI supera 0.25 en cualquier mes, el sistema SHALL destacar ese punto en el gráfico con marcador de alerta.
- [EARS-DB06] El sistema SHALL permitir filtrar la cartera por arquetipo de cliente (good/recurrent/over/fraud/low) para análisis segmentado.
- [EARS-DB07] El sistema SHALL mostrar simulación contrafactual: trayectoria con vs. sin control LQR para un cliente seleccionable.

### Infraestructura y Test
#### Base de datos

- [EARS-I01] El sistema SHALL crear las tablas SQLite: customers, monthly_states, estimated_states, decisions, metrics en db/credito.db.
- [EARS-I02] WHEN se persiste un estado estimado, el sistema SHALL guardar x_hat (4 floats), traza de P (float), y score_dinamico (float).
- [EARS-I03] El sistema SHALL usar un repositorio por agregado de dominio (CustomerRepository, DecisionRepository).
- [EARS-I04] IF db/credito.db no existe, el sistema SHALL crearla e inicializar el esquema en el arranque.

#### Tests

- [EARS-T01] El sistema SHALL incluir tests pytest para: pipeline de features, identificación de matrices, filtro de Kalman, controlador LQR, modelo logístico.
- [EARS-T02] WHEN se ejecuta pytest tests/, el sistema SHALL completar sin errores con cobertura ≥ 80% de los módulos en modelos/.
- [EARS-T03] El sistema SHALL incluir test de integración que ejecute el pipeline completo con los 500 clientes y verifique métricas mínimas.
- [EARS-T04] IF cualquier test de Kalman produce covarianza con autovalores negativos, el test SHALL fallar explícitamente con mensaje descriptivo.


### Mapa de dependencias entre fases

- [EARS-DEP01] La Fase 2 SHALL depender de que db/raw_transactions.csv y db/features_dinamicos.csv existan (salida de Fase 1).
- [EARS-DEP02] La Fase 3 SHALL depender de db/modelo_logistico.pkl y db/matrices_sistema.npz (salidas de Fases 1 y 2).
- [EARS-DEP03] La Fase 4 SHALL depender de db/credito.db populado con estados estimados y decisiones (salida de Fase 3).
- [EARS-DEP04] IF cualquier dependencia de fase no existe, el sistema SHALL lanzar FileNotFoundError con ruta esperada y fase responsable.

### Tabla de Trazabilidad AGENTS.md → EARS

| Sección AGENTS.md | Requisitos EARS cubiertos |
|---|---|
| Planta del cliente (x_c, A, B) | EARS-S01, S02, S03 |
| Observador Kalman | EARS-K01..K05 |
| Controlador LQR | EARS-C01..C06 |
| Modelo logístico baseline | EARS-B01..B06 |
| Feature engineering | EARS-F01..F06 |
| Dashboard cliente | EARS-DC01..DC06 |
| Dashboard banco | EARS-DB01..DB07 |
| DDD + código limpio | EARS-G01..G08 |
| Tests pytest | EARS-T01..T04 |
| Fases 1-4 | EARS-DEP01..DEP04 |

---

## Stack de tecnología:

- Python (3.12)
- Scikit-learn (para el modelo logístico clásico)
- Python-control (para LQR y el observador)
- NumPy
- Pandas
- Streamlit
- SQLite
- Aplicación monolítica

## Preferencias generales:

- Comunicación del agente en español.


## Preferencia de Código:

- Usar el enfoque DDD (Domain-Driven Design).
- Hojas de estilo globales/locales, inyectando archivos CSS personalizados para clases globales según cada parte (cliente y banco). Cada parte (cliente y banco) deben de tener su propia carpeta donde estarían los archivos CSS personalizados, donde al mismo nivel que esas carpetas habría un CSS global.
- Usar buenas prácitcas en NiceGUI
- Cualquier función no deben superar las 60 líneas.
- Se debe de declarar siempre el tipo de cada variable
- Densidad de Aserciones: Al menos dos aserciones por función para verificar invariantes y precondiciones.
- El modelo de regresión solo se debe de entrenar una sola vez, por lo que, se debe de guardar en .pkl para su posterior uso.
- Prioriza el código legible y mantenible.
- Prioriza que el código sea sencillo de entender y con el mínimo de comentarios.


---

## Estructura de archivos:
- carpeta "db" (donde se guarda el sqlite, el .csv del raw de las transacciones de los clientes y el .pkl del modelo de regresión)
- AGENTS.md
- carpeta "modelos" (donde estara el código para la gestión del modelo estático y dinámico, aquí usa la estructura de archivos más adecuadas para esta parte)
- carpeta "gui" (donde se guardara el código para la interfaz gráfica para la visualización de la parte del cliente y el banco, aquí se utiliza la estructura de archivos en DDD)
- carpeta "tests" (donde se guardaran los test de código[pytest], aserciones y comparaciones entre modelos)
- carpeta "docs" (donde se guardan documentación relevante del proyecto)
- carpeta "utils" (código generico o repetitivo que es general, no importante para el proyecto pero si para no repetir código como por ejemplo la lectura de un archivo)

---

## Fases de desarrollo:

1. Fase 1 — Dataset y features dinámicos:
    - Generar datos sintéticos de clientes con trayectorias temporales realistas (ventana de 24 meses).
    - Construir features temporales: ratio deuda/ingreso promedio móvil, tendencia de utilización, volatilidad de pagos.
2. Fase 2 — Modelo estático:
    - Baseline: modelo logístico clásico (benchmark).
3. Fase 3 — Modelo dinámico:
    - Identificar matrices $A$, $B$, $C$ del sistema cliente por regresión sobre datos históricos.
    - Implementar filtro de Kalman para estimar estado del cliente con observaciones parciales.
    - Ganancia de control $K$ por LQR o por optimización directa de la pérdida crediticia
4. Fase 4 — Comparación y validación:
    - Comparar con scorecard estático en simulación: ¿el modelo dinámico reduce pérdidas en backtesting?
    - Cálculo de métricas: Gini, KS, pérdida esperada de cartera, estabilidad del score en el tiempo (PSI).
5. Fase 5 — Streamlit:
    - Streamlit: dashboard de análisis de datos de un nuevo dataframe (que se guardara en memoria RAM).
    - Streamlit: dashboard donde el modelo estático evaluara el dataframe del punto anterior.
    - Streamlit: dashboard donde el modelo dinámico evaluara el dataframe guardado en memoria RAM.
    - Streamlit: dashboard donde se comparara ambos modelos.

---

## Otras consideraciones:

- Guarda y revisa el plan de implementación en un fichero en "docs/plan-implementacion.md".
- Guarda y revisa las tareas y su estado en un fichero en "docs/tareas.md".

---

## Modo implementación:

- Sólo código, mínimos comentarios, el código ya debe de ser autoexplicativo.
- No expliques que hace el código en el chat del agente.
- Responde en español si preguntas, pero en prompts internos usa inglés.
- Si hay ambiguedad, pregunta indicando la opción más simple y que no rompa nada.