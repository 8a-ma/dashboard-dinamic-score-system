import pandas as pd
from settings.settings import settings
from modelos.evaulacion.factory import SimulatorFactory
from modelos.evaulacion.metrics import MetricsCalculator
from modelos.evaulacion.repository import IBacktestingRepository


class BacktestingOrchestrator:
    def __init__(self, factory: SimulatorFactory, repository: IBacktestingRepository, metrics_calculator: MetricsCalculator):
        self.factory = factory
        self.repository = repository
        self.metrics_calculator = metrics_calculator

    def run(self, df: pd.DataFrame) -> dict:
        log_results = self.factory.create_logistic().simulate(df)
        dyn_results = self.factory.create_dynamic().simulate(df)

        comparison = self.metrics_calculator.compare(log_results, dyn_results)

        self.repository.save_comparison_json(comparison)

        return comparison