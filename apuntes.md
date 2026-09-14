- Arquitectura general: ...
    - Parte de cliente: Aspectos que vería el cliente del modelo implementado
    - Parte banco: Aspectos que vería el banco
- Funcionalidades de la aplicación: Especificaciones técnicas usando la notación EARS
- Stack de tecnología: Tecnologías usadas en el proyecto.
- Preferencias generales: Notas para que el agente tenga en cuenta


Crea un plan de implementación detallado para que un agente de ia sea capaz de desarrollar el proyecto y todas las funcionalidades descritas en el fichero AGENTS.md . NO crees código, solo crea el plan de implementación completo.

Siguiendo `AGENTS.md` y `docs/plan-implementación.md`. Necesito que revises y corrijas el código de `modelos/features/pipeline.py`. Las funciones y las variables deben de estar en inglés

Siguiendo `docs/plan-implementación.md`. Necesito que revises y corrijas el código de `modelos/evaluacion/backtesting.py`. Las funciones y las variables deben de estar en inglés en lo posible.

Refactoriza el siguiente código usando un patrón de diseño (usando POO) enfocado en el entrenamiento, guardado y carga del modelo. La función `initialize` se mantiene como función separada de la nueva clase, esta función se le debe de agregar la opción de entrenar un nuevo modelo (reemplazando el existente) o seguir con la lógica que ya se tiene (si el modelo existe entonces solo se carga, sino se entrena y guarda)



CLASE BacktestingOrchestrator:

    MÉTODO run(df, shock=None):
        # Precondición: df no vacío con columnas del dominio
        # Precondición: shock es None o dict válido

        log_sim = factory.create_logistic()
        dyn_sim = factory.create_dynamic()

        log_base = log_sim.simulate(df, shock=None)
        dyn_base = dyn_sim.simulate(df, shock=None)

        comparison_base = metrics_calculator.compare(log_base, dyn_base)

        SI shock NO ES None:
            log_shock = log_sim.simulate(df, shock=shock)
            dyn_shock = dyn_sim.simulate(df, shock=shock)

            comparison_shock = metrics_calculator.compare(log_shock, dyn_shock)

            comparison_base['escenario_shock'] = {
                'logistico': {
                    'perdida_total':         comparison_shock['logistico']['perdida_total'],
                    'delta_perdida':         comparison_shock['logistico']['perdida_total']
                                             - comparison_base['logistico']['perdida_total'],
                    'tasa_mora_mensual':     comparison_shock['logistico']['tasa_mora_mensual'],
                    'verdaderos_rechazos':   comparison_shock['logistico']['verdaderos_rechazos'],
                },
                'dinamico': {
                    'perdida_total':         comparison_shock['dinamico']['perdida_total'],
                    'delta_perdida':         comparison_shock['dinamico']['perdida_total']
                                             - comparison_base['dinamico']['perdida_total'],
                    'tasa_mora_mensual':     comparison_shock['dinamico']['tasa_mora_mensual'],
                    'verdaderos_rechazos':   comparison_shock['dinamico']['verdaderos_rechazos'],
                },
                'resiliencia_diferencial':   delta_perdida_logistico - delta_perdida_dinamico
                    # positivo → dinámico absorbió mejor el shock
            }

        repository.save_comparison_json(comparison_base)
        persistir en SQLite: log_base, dyn_base (estados, decisiones)

        RETORNAR comparison_base

