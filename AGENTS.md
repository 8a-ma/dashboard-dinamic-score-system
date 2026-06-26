# Proyecto: dashboard-dinamic-score-system

## Rol del agente:
...

---

## Problema que se resuelve:

Los modelos de scoring tradicionales (regresión logística, scorecard) son estáticos: producen una probabilidad de default en un instante dado, sin considerar que el comportamiento del cliente cambia en función de las decisiones del banco (si le doy más crédito, su comportamiento cambia). Un banco que usa un score estático puede crear incentivos perversos: dar crédito a clientes que precisamente por recibirlo van a sobre-endeudarse.


## Objetivo general:

Construir un motor donde el score sea la salida de un sistema dinámico que modela la evolución del cliente y se ajusta con las decisiones del banco. El motor utiliza aspectos de la ingenieria de control utilizando lazo de control realimentado con estado estimado (observador):

- Planta: el cliente es un sistema con estado $x_c = [deuda, ingreso, utilización, días mora]^T$ que evoluciona según $\dot{x}_c=Ax_c+Bu_b$ donde $u_b$ es la política del banco (límite de crédito).
- Observador (Kalman): el banco no observa el estado completo (no sabe el ingreso exacto). Con las señales disponibles (transacciones, pagos), estima $\hat{x}_c$.
- Controlador: la decisión de aprobar/rechazar/ajustar límite es $u_b = K\hat{x}_c$ donde $K$ se diseña para minimizar pérdida esperada (análogo a LQR con Q = costo de default, R = costo de capital comprometido).
- Realimentación: la decisión $u_b$ afecta al estado del cliente → el loop se cierra.


## Consideraciones generales:

Estas reglas aplican SIEMPRE a todas las fases y funcionalidades:

- Priorizar sencillez y de fácil entendimiento.
- Priorizar buenas prácticas y seguridad.
- Priorizar reutilización de clases/funciones.


## Arquitectura general:

### Parte de cliente:

La parte del cliente incluirá:

- Su propio dashboard con las transferencias realizadas, además de mostrar trayectoria de estado estimado y decisiones recomendadas.

### Parte de banco:

La parte del banco incluirá:

- Un dashboard que muestre features temporales (ratio deuda/ingreso promedio móvil, tendencia de utilización, volatilidad de pagos).
- Baseline: modelo logístico clásico (benchmark).
- Una sección para visualizar el modelo dinámico.
- Comparación con scorecard estático en simulación con el modelo dinámico
- Visualización de métricas: Gini, KS, pérdida esperada de cartera, estabilidad del score en el tiempo (PSI).

---

## Funcionalidades de la aplicación:

DISEÑAR CADA SPEC

---

## Stack de tecnología:

- Python
- Scikit-learn (para el modelo logístico clásico)
- Python-control (para LQR y el observador)
- NumPy
- Pandas
- NiceGUI (para interfaz gráfica)
- SQLite
- Aplicación monolítica

## Preferencias generales:

- Comunicación del agente en español.
- Los dashboard para cada parte (cliente, banco) deben ser visualmente agradable, sencillos de entender y muy intuitivo.

## Preferencias de diseño para la parte del cliente:

- Colores: {
  --color-1: #3d8a5c;
  --color-2: #6ebf7f;
  --color-3: #9fd5a6;
  --color-4: #d0e7d0;
  --color-5: #f5faf5;
}

## Preferencias de diseño para la parte del banco:

- Colores: {
  --color-1: #2e3f52;
  --color-2: #4a6c8c;
  --color-3: #6db0d0;
  --color-4: #a0d8e9;
  --color-5: #f4e1a4;
}

## Preferencias de estilo:

- Uso de medidas en rem, usando un font-size base de 10px
- Uso de CSS module, dando estilos exclusivos a cada parte (cliente y banco) y sus propios componentes.
- Uso de CSS3 nativo.
- Uso de buenas prácticas de maquetación css en NiceGUI, si es necesario usa flexbox y css grid layout.

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

---

## Fases de desarrollo:

1. Fase 1 — Dataset y features dinámicos:
    - Generar datos sintéticos de clientes con trayectorias temporales realistas (ventana de 24 meses).
    - Construir features temporales: ratio deuda/ingreso promedio móvil, tendencia de utilización, volatilidad de pagos.
    - Baseline: modelo logístico clásico (benchmark).
2. Fase 2 — Modelo dinámico:
    - Identificar matrices $A$, $B$, $C$ del sistema cliente por regresión sobre datos históricos.
    - Implementar filtro de Kalman para estimar estado del cliente con observaciones parciales.
    - Ganancia de control $K$ por LQR o por optimización directa de la pérdida crediticia
3. Fase 3 — Comparación y validación:
    - Comparar con scorecard estático en simulación: ¿el modelo dinámico reduce pérdidas en backtesting?
    - Cálculo de métricas: Gini, KS, pérdida esperada de cartera, estabilidad del score en el tiempo (PSI).
4. Fase 4 — Interfaz:
    - NiceGUI: dashboard de un cliente individual, mostrando trayectoria de estado estimado y decisiones recomendadas.
    - NiceGUI: dashboard del banco.

---

## Otras consideraciones:

- Guarda y revisa el plan de implementación en un fichero en "docs/plan-implementacion.md".
- Guarda y revisa las tareas y su estado en un fichero en "docs/tareas.md".

---

## Modo implementación:

- Sólo código, mínimos comentarios, el código ya debe de ser autoexplicativo.
- No expliques que hace el código en el chat del agente.
- Responde en español si preguntas, pero en prompts internos usa inglés / español libremente.
- Si hay ambiguedad, pregunta indicando la opción más simple y que no rompa nada.