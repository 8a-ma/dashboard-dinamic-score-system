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

| ID | Patrón EARS | Enunciado del Requisito (Requirement Statement) | Agente / Módulo Responsable | Método de Validación |
| :--- | :---: | :--- | :---: | :--- |
| **EARS-G01** | Ubiquitous | El sistema **SHALL** ejecutarse en su totalidad como una aplicación monolítica escrita bajo Python 3.12. | Arquitectura de Software | Inspección del Entorno de Ejecución |
| **EARS-G02** | Ubiquitous | El sistema **SHALL** persistir y recuperar todos los artefactos persistentes (SQLite, CSV, PKL) exclusivamente dentro de la ruta `db/`. | Persistencia / Infraestructura | Inspección física del Directorio de Datos |
| **EARS-G03** | Ubiquitous | El sistema **SHALL** estructurarse siguiendo los principios de diseño guiado por el dominio (DDD), separando el software en las capas de dominio, aplicación e infraestructura dentro de `gui/`. | Arquitectura de Software | Revisión de Código y Estructura de Directorios |
| **EARS-G04** | While | **WHILE** se implemente cualquier función o método de la solución, el sistema **SHALL** garantizar que su longitud física no exceda en ningún caso de 60 líneas de código. | Ingeniero de Software / QA | Análisis Estático de Código (AST / Flake8) |
| **EARS-G05** | Ubiquitous | El sistema **SHALL** declarar anotaciones de tipo estático explícitas (`Type Hinting`) en todos los parámetros de entrada y retorno de funciones o métodos. | Ingeniero de Software / QA | Verificación estática con MyPy |
| **EARS-G06** | Ubiquitous | El sistema **SHALL** incluir como mínimo dos cláusulas de aserción (`assert`) por función para validar de forma robusta las precondiciones, invariantes o postcondiciones de ejecución. | Ingeniero de QA / Programador | Ejecución de Suite de Pruebas Unitarias |
| **EARS-G07** | If | **IF** el archivo serializado `db/modelo_logistico.pkl` no existe en disco, el sistema **SHALL** entrenar automáticamente un nuevo clasificador estático de regresión logística y guardarlo en el directorio de persistencia antes de realizar predicciones. | Agente de Modelo Estático | Prueba de Integración de Inicialización |
| **EARS-G08** | Where | **WHERE** se construya o modifique cualquier módulo de software, el sistema **SHALL** priorizar de manera absoluta la legibilidad, mantenibilidad y el diseño autoexplicativo por sobre optimizaciones prematuras de performance. | Arquitecto de Software | Revisión por pares (Peer Review) |
| **EARS-G09** | If | **IF** alguna de las dependencias u objetos persistidos en disco requeridos para la ejecución de una fase no se localiza en la ruta esperada, el sistema **SHALL** lanzar una excepción del tipo `FileNotFoundError` detallando el recurso ausente y el componente causante. | Capa de Infraestructura / DB | Pruebas de Integración y Fallo Controlado |

### Fase 1 Generación de datos sintéticos (`scripts/transacciones.py`)

Esta fase modela la simulación de la planta del cliente, su comportamiento financiero dinámico, shocks externos y dependencias temporales.

| ID | Patrón EARS | Enunciado del Requisito (Requirement Statement) | Agente / Módulo Responsable | Método de Validación |
| :--- | :---: | :--- | :---: | :--- |
| **EARS-F1-01** | Ubiquitous | El script `scripts/transacciones.py` **SHALL** generar un entorno de datos sintéticos compuesto de exactamente 500 clientes únicos con un historial temporal continuo de 24 meses. | Generador de Datos / Builder | Validación de registros e IDs únicos en DB |
| **EARS-F1-02** | Ubiquitous | El sistema **SHALL** asignar a cada uno de los 500 clientes una clasificación fija y exclusiva en exactamente uno de los 5 arquetipos crediticios con su respectiva distribución estadística: `good` (40%), `recurrent` (25%), `over` (20%), `fraud` (5%), y `low` (10%). | Generador de Datos / Builder | Prueba estadística de bondad de ajuste Chi-cuadrado |
| **EARS-F1-03** | Ubiquitous | El simulador **SHALL** generar exactamente 24 observaciones mensuales secuenciales por cliente, de modo que el dataset contenga de manera determinista 12,000 registros históricos. | Generador de Datos | Conteo de filas agregadas por cliente |
| **EARS-F1-04** | Ubiquitous | El simulador **SHALL** incorporar dependencia temporal para modelar la correlación secuencial en los indicadores financieros de un mismo cliente mes a mes basándose en modelos autorregresivos. | Generador de Datos | Test de Autocorrelación de Durbin-Watson |
| **EARS-F1-05** | Ubiquitous | El sistema **SHALL** modelar un lazo de retroalimentación temporal autoregresiva donde la situación financiera del cliente tienda a mejorar, mantenerse estable o deteriorarse según su historial de endeudamiento reciente. | Generador de Datos / Planta | Simulación de transiciones y análisis de trayectorias |
| **EARS-F1-06** | Ubiquitous | El generador de datos **SHALL** inyectar shocks financieros exógenos imprevistos con una probabilidad de ocurrencia controlada estrictamente de entre el 1% y el 5% por mes. | Generador de Datos / Planta | Simulación de Monte Carlo e inspección de outliers |
| **EARS-F1-07** | Ubiquitous | El sistema **SHALL** aplicar un proceso de desplazamiento de comportamiento (`drift`) sobre el segmento crediticio de un cliente dentro de un intervalo temporal aleatorio definido. | Generador de Datos / Segmentos | Test de detección de derivas en la distribución (KS-Test) |
| **EARS-F1-08** | Ubiquitous | El sistema **SHALL** inyectar ruido estocástico aditivo representativo de eventos extraordinarios de un único mes, tales como ingresos de capital imprevistos dirigidos a mitigar un porcentaje aleatorio de la deuda acumulada. | Generador de Datos / Ruido | Inspección de perfiles individuales y shocks |
| **EARS-F1-09** | When | **WHEN** la rutina de generación de datos y simulación temporal culmina exitosamente, el sistema **SHALL** escribir y sobrescribir el archivo consolidado `db/raw_transactions.csv`. | Infraestructura / Escritor CSV | Test de existencia y consistencia de archivo en disco |
| **EARS-F1-10** | Ubiquitous | El pipeline de ingeniería de características (feature engineering) **SHALL** calcular y consolidar características temporales que incluyan: ratio de deuda/ingreso promedio móvil, tendencia de utilización de línea de crédito y volatilidad agregada de pagos. | Ingeniería de Características | Test de cobertura y completitud de variables |

### Fase 2 Modelo Estático (Baseline de Regresión Logística)
Especificación del modelo tradicional de riesgo que actúa como benchmark estático en lazo abierto.

| ID | Patrón EARS | Enunciado del Requisito (Requirement Statement) | Agente / Módulo Responsable | Método de Validación |
| :--- | :---: | :--- | :---: | :--- |
| **EARS-F2-01** | Ubiquitous | El pipeline del modelo estático **SHALL** implementar una secuencia automatizada y reproducible que reciba un dataframe procesado, entrene un clasificador de regresión logística estándar y guarde el modelo serializado. | Agente de Modelo Estático | Ejecución del pipeline de entrenamiento |
| **EARS-F2-02** | If | **IF** el archivo serializado `db/modelo_logistico.pkl` ya se localiza en disco, el sistema **SHALL** omitir el proceso de entrenamiento y cargarlo directamente a la memoria activa. | Modelo Estático / Infraestructura | Medición del tiempo de carga inicial de servicios |
| **EARS-F2-03** | When | **WHEN** culmina el entrenamiento de un modelo estático, el sistema **SHALL** generar y guardar un reporte JSON documentando las métricas de clasificación tradicionales (AUC-ROC, Gini, KS, PSI). | Modelo Estático / Evaluador | Verificación del archivo de salida JSON generado |
| **EARS-F2-04** | Ubiquitous | El clasificador estático **SHALL** estimar la probabilidad de default de forma puntual e instantánea, emitiendo una probabilidad que asume condiciones de lazo abierto e ignora el impacto del límite de crédito sobre el comportamiento de pago futuro del cliente. | Agente de Modelo Estático | Análisis de insensibilidad del score ante la acción de control |

### Fase 3 Modelo Dinámico (Sistema en Espacio de Estados, Filtro de Kalman y LQR)
Requisitos para el núcleo matemático del motor de control dinámico.

| ID | Patrón EARS | Enunciado del Requisito (Requirement Statement) | Agente / Módulo Responsable | Método de Validación |
| :--- | :---: | :--- | :---: | :--- |
| **EARS-F3-01** | Ubiquitous | El sistema **SHALL** estimar los coeficientes y matrices $A$, $B$ y $C$ correspondientes al modelo dinámico en espacio de estados aplicando regresión multivariable sobre los datos históricos procesados. | Agente de Control / Estimador | Pruebas de significación de regresión multivariante |
| **EARS-F3-02** | Ubiquitous | El sistema **SHALL** modelar el vector de estados ocultos del cliente mediante la especificación $x_c = [deuda, ingreso, utilización]^T$ y modelar la acción de control del banco $u_b$ como el límite de crédito disponible. | Agente de Control / Planta | Verificación dimensional e inspección matemática |
| **EARS-F3-03** | Ubiquitous | El sistema **SHALL** implementar un filtro de Kalman de tiempo discreto como observador de estados latentes para estimar $\hat{x}_c$ recursivamente a partir de pagos y volumen de transacciones con ruido de medición. | Observador (Filtro de Kalman) | Evaluación del error de estimación cuadrático medio |
| **EARS-F3-04** | Ubiquitous | El sistema **SHALL** sintetizar de forma óptima la matriz de ganancias de control $K$ a través de un regulador cuadrático lineal (LQR), sintonizando las matrices de costo $Q$ (pérdida esperada por impago) y $R$ (costo de capital). | Controlador (LQR) | Simulación teórica y análisis de autovalores de lazo cerrado |
| **EARS-F3-05** | Ubiquitous | El modelo de lazo cerrado **SHALL** calcular la acción de control de límite de crédito dinámico de manera determinista mediante la expresión de realimentación $u_b = -K\hat{x}_c$. | Controlador (LQR) / Planta | Simulación en tiempo de ejecución del lazo cerrado |
| **EARS-F3-06** | If | **IF** ya se dispone de las matrices estimadas del modelo dinámico en persistencia local, el sistema **SHALL** cargarlas de manera instantánea en los objetos controladores de memoria en lugar de recalcularlas. | Agente de Control / Cache | Prueba de carga de configuración dinámica |
| **EARS-F3-07** | When | **WHEN** el ciclo de identificación dinámica se completa, el sistema **SHALL** generar y guardar un reporte JSON que contenga las métricas de estabilidad del sistema y errores residuales del observador. | Agente de Control / Evaluador | Verificación del esquema JSON de calibración |

### Fase 4 Comparación, Validación y Simulación Temporal
Requisitos enfocados en el backtesting secuencial y la comparación equitativa entre políticas de lazo abierto y lazo cerrado.

| ID | Patrón EARS | Enunciado del Requisito (Requirement Statement) | Agente / Módulo Responsable | Método de Validación |
| :--- | :---: | :--- | :---: | :--- |
| **EARS-F4-01** | Ubiquitous | El módulo comparador **SHALL** calcular y contrastar de manera equivalente las métricas de discriminación y performance estadística (AUC-ROC, coeficiente de Gini, estadística KS y PSI) de ambos enfoques. | QA / Módulo Evaluador | Validación del pipeline matemático y consistencia de datos |
| **EARS-F4-02** | Ubiquitous | El módulo de evaluación financiera **SHALL** computar métricas clave de negocio que incluyan: pérdida esperada acumulada, tasa de morosidad agregada y variabilidad temporal del límite de crédito otorgado. | QA / Negocio Crediticio | Simulación económica de cartera |
| **EARS-F4-03** | Ubiquitous | El evaluador **SHALL** ejecutar una simulación temporal secuencial paso a paso (mes a mes) utilizando los datos retenidos en la memoria RAM para emular la dinámica evolutiva de los clientes bajo ambas políticas de crédito de forma paralela. | Módulo de Backtesting | Simulación secuencial (Time-step simulation) |
| **EARS-F4-04** | Where | **WHERE** se apliquen escenarios estresados con shocks exógenos extremos en el simulador, el sistema **SHALL** comparar la resiliencia agregada e incremental en defaults financieros de ambos modelos de scoring. | Módulo de Backtesting / QA | Pruebas de robustez y estrés financiero |

### Fase 5 Revisión de Código y Patrones de Diseño
Implementación del control de calidad por diseño y directivas de los agentes inteligentes.

| ID | Patrón EARS | Enunciado del Requisito (Requirement Statement) | Agente / Módulo Responsable | Método de Validación |
| :--- | :---: | :--- | :---: | :--- |
| **EARS-F5-01** | Ubiquitous | El sistema **SHALL** forzar que el agente de desarrollo de control de crédito implemente la creación de trayectorias financieras e instancias de clientes utilizando de forma mandatoria el patrón **Builder**. | Patrones de Diseño / Builder | Inspección de la definición de clases y dependencias |
| **EARS-F5-02** | Ubiquitous | El código del sistema **SHALL** estructurarse utilizando herencia de interfaces limpias, inyección de dependencias y desacoplamiento estructural, mitigando la duplicación bajo el principio DRY. | Arquitectura de Software | Análisis de cohesión de código y métricas de duplicidad |
| **EARS-F5-03** | Ubiquitous | El proyecto **SHALL** incorporar el archivo de gobernanza de agentes inteligentes `AGENTS.md` en el directorio raíz de la aplicación para establecer los límites y deudas técnicas de cada rol de desarrollo. | Gobernanza de Agentes IA | Inspección visual de la raíz del proyecto |
| **EARS-F5-04** | While | **WHILE** los agentes inteligentes sugieran optimizaciones o refactorizaciones, el prompt de control **SHALL** restringir sugerencias de código a aquellas con una longitud máxima por bloque menor a 60 líneas y con tipado estricto. | Ingeniería de Prompts / QA | Validación automatizada de prompts del agente |
| **EARS-F5-05** | Ubiquitous | El sistema de desarrollo **SHALL** aislar de forma modular las reglas de dominio del simulador financiero de las rutinas de infraestructura gráfica de presentación visual. | Arquitectura de Software / DDD | Prueba de importación de módulos (cero dependencias de UI en Core) |

### Fase 6 Requisitos de la capa de visualización e interacción con el usuario de negocio e ingeniería.
Requisitos de la capa de visualización e interacción con el usuario de negocio e ingeniería.

| ID | Patrón EARS | Enunciado del Requisito (Requirement Statement) | Agente / Módulo Responsable | Método de Validación |
| :--- | :---: | :--- | :---: | :--- |
| **EARS-F6-01** | Ubiquitous | El dashboard interactivo de monitoreo y control **SHALL** implementarse utilizando Streamlit (Python 3.12) como interfaz web unificada de presentación. | Ingeniero de Software / GUI | Ejecución y renderizado correcto de la app Streamlit |bl |
| **EARS-F6-03** | Ubiquitous | El dashboard **SHALL** proporcionar una pestaña de *Análisis Descriptivo y Diagnóstico* para evaluar un dataframe completamente nuevo (almacenado en RAM), desplegando resúmenes estadísticos (distribución, media, varianza) y alertas de diagnóstico (correlaciones extremas, valores atípicos y drift de segmentos). | Ingeniero de Software / Módulo EDA | Pruebas de carga de dataframes sintéticos alternativos en RAM |
| **EARS-F6-04** | Ubiquitous | El dashboard **SHALL** renderizar una pestaña para el *Modelo Estático* que proyecte su performance predictiva histórica (curva AUC-ROC, gráfico de estadística KS, coeficiente de Gini y PSI) y un gráfico de dispersión de asignación de límites de crédito. | Ingeniero de Software / Módulo Estático | Validación visual del pipeline de evaluación estática |
| **EARS-F6-05** | Ubiquitous | El dashboard **SHALL** renderizar una pestaña interactiva para el *Modelo Dinámico* que grafique la convergencia temporal de los estados estimados por el filtro de Kalman ($\hat{x}_c = [deuda, ingreso, utilización]^T$) versus sus valores reales, junto al perfil histórico de las acciones de control de límite de crédito ($u_b$). | Ingeniero de Control / Módulo Dinámico | Verificación visual de simulación de estados y límites |
| **EARS-F6-06** | Ubiquitous | El dashboard **SHALL** proporcionar un *Dashboard Comparativo de Negocio* que contraste las métricas financieras agregadas (pérdida esperada acumulada, tasa de morosidad y estabilidad de la línea de crédito) entre ambas políticas de scoring. | Ingeniero de QA / Módulo Comparación | Consistencia analítica de indicadores de negocio |
| **EARS-F6-07** | Ubiquitous | El dashboard comparativo **SHALL** ejecutar una simulación secuencial paso a paso (mes a mes) a lo largo de un horizonte empírico de 24 meses para calcular las pérdidas esperadas e indicar de forma discreta la decisión de crédito ("Préstamo Aprobado" o "Préstamo Rechazado") para cada cliente bajo ambos modelos. | Ingeniero de Control / Módulo Comparación | Validación del bucle de simulación temporal paso a paso |
| **EARS-F6-08** | If | **IF** el usuario activa un shock macroeconómico o financiero extremo en la simulación temporal del dashboard comparativo, el sistema **SHALL** recalcular dinámicamente y graficar la variación de la pérdida esperada y la respuesta de mora acumulada para ambos enfoques. | Ingeniero de QA / Módulo Comparación | Test de resiliencia ante shocks interactivos en la GUI |
| **EARS-F6-09** | When | **WHEN** se renderiza cualquier componente interactivo que requiera procesamiento de datos de entrenamiento o simulación, el sistema **SHALL** implementar al menos dos aserciones (`assert`) por función para verificar la existencia del dataframe en memoria y la integridad de sus columnas esenciales antes de graficar. | Ingeniero de QA / Linter | Cobertura de pruebas unitarias de UI y ejecución de linter |

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
- carpeta "gui" (donde se guardara el código para la interfaz gráfica para la visualización del análisis de un cartera de clientes nueva, visualización del rendimiento del modelo estatico, visualización del modelo de rendimiento dinámico y una comparación entre ambos modelos, aquí se utiliza la estructura de archivos en DDD)
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
    - Se revisara el código de las fases anteriores con el fin de optimizarlo y de mejorarlo.
    - Se aplicaran los patrones de diseño de software donde corresponda (por ejemplo: en la creación del raw de clientes, se puede hacer una clase Cliente usando el patrón builder).
    - Los patrones de diseño deben corresponder al problema que se esta resolviendo en el código actual.
    - La optimización debe estar enfocada en evitar el overhead
    - Se debe de implementar logs en el código, siguiendo la estructura canonica ([TIMESTAMP] | [LEVEL] | [SERVICE / MODULE] | [TRACE ID] | MESSAGE)
    - La configuración del log se hace instancia una sola vez. Las demás clases y funciones toman esa instancia y llaman al método que corresponda

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
- El código debe enfocarse primero en POO luego en paradigma funcional y luego en el paradigma procedural
- No expliques que hace el código en el chat del agente.
- Responde en español si preguntas, pero en prompts internos usa inglés.
- Si hay ambiguedad, pregunta indicando la opción más simple y que no rompa nada.