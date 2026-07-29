import sqlite3
import pandas as pd
from pathlib import Path
from abc import ABC, abstractmethod
from settings.settings import settings
from utils.file_helpers import save_dict_to_json


class IBacktestingRepository(ABC):
    @abstractmethod
    def save_monthly_states(self, conn: sqlite3.Connection, df: pd.DataFrame) -> None: pass

    @abstractmethod
    def save_estimates_and_decisions(self, conn, dynamic_df: pd.DataFrame, logistic_df: pd.DataFrame) -> None: pass

    @abstractmethod
    def save_comparison_json(self, comparison: dict) -> None: pass


class SQLiteBacktestingRepository(IBacktestingRepository):
    def save_comparison_json(self, comparison: dict) -> None:
        save_dict_to_json(comparison, settings.COMPARISON_PATH)

    def save_estimates_and_decisions(self, conn, dynamic_df, logistic_df):
        pass

    def save_monthly_states(self, conn, df):
        pass
