import pandas as pd
from modelos.evaulacion.factory import SimulatorFactory
from modelos.evaulacion.metrics import MetricsCalculator
from modelos.evaulacion.repository import SQLiteBacktestingRepository
from modelos.evaulacion.orchestrator import BacktestingOrchestrator



def run_backtesting(df: pd.DataFrame) -> dict:
    orchestrator = BacktestingOrchestrator(
        SimulatorFactory(),
        SQLiteBacktestingRepository(),
        MetricsCalculator()
    )

    return orchestrator.run(df)