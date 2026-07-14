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

| ID | Patrón EARS | Requisito |
|---|---|---|
| EARS-G01 | Ubiquitous | El sistema SHALL ejecutarse como aplicación monolítica en Python 3.12. |
| EARS-G02 | Ubiquitous | El sistema SHALL persistir todos los artefactos generados en la carpeta `db/`. |
| EARS-G03 | Ubiquitous | El sistema SHALL seguir arquitectura DDD separando dominio, aplicación e infraestructura dentro de `gui/`. |
| EARS-G04 | While | WHILE cualquier función es implementada, el sistema SHALL garantizar que su longitud no supere 60 líneas. |
| EARS-G05 | Ubiquitous | El sistema SHALL declarar tipo explícito en cada variable y parámetro de función. |
| EARS-G06 | Ubiquitous | El sistema SHALL incluir al menos dos aserciones por función para verificar precondiciones e invariantes. |
| EARS-G07 | If | IF `db/modelo_logistico.pkl` no existe, el sistema SHALL entrenarlo y persistirlo antes de cualquier predicción. |
| EARS-G08 | Where | WHERE se implementa cualquier módulo, el sistema SHALL priorizar legibilidad sobre optimización prematura. |
| EARS-G09 | If | IF cualquier dependencia de fase no existe en disco, el sistema SHALL lanzar `FileNotFoundError` indicando la ruta esperada y la fase responsable. |

### Fase 1
#### 1.1 Generación de datos sintéticos (`scripts/transacciones.py`)

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
- Usar buenas prácitcas en Streamlit
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

5. Fase 5 — Revisión de código:
    - Se revisara el código de las fases anteriores con el fin de optimizarlo y de mejorarlo
    - Se aplicaran los patrones de diseño de software donde se pueda (por ejemplo en la creación del raw de clientes, se puede hacer una clase Cliente usando el patrón builder)

6. Fase 6 — Streamlit:
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