import numpy as np
import pandas as pd
from settings.settings import settings
from modelos.dominio.arquetipos import ARCHETYPE_COMPOSITION
from modelos.dominio.customer_builder import CustomerBuilder


class CustomerDirector:
    def __init__(self, builder: CustomerBuilder):
        self._builder = builder
    
    def _calculate_composition(self, n: int) -> dict[str, int]:
        composition: dict[str, int] = {}

        for key, value in ARCHETYPE_COMPOSITION.items():
            composition[key] = int(value * n)

        remainder: int = n - sum(composition.values())

        composition[list(composition.keys())[-1]] += remainder

        assert sum(composition.values()) == n

        return composition
    
    def build_dataset(self, seed: int = 42) -> pd.DataFrame:
        assert seed > 0
        np.random.seed(seed)

        composition: dict[str, int] = self._calculate_composition(settings.N_CUSTOMERS)
        types: list[str] = []

        for type, quant in composition.items():
            types += [type] * quant
        
        np.random.shuffle(types)

        rows: list[dict[str, object]] = []

        for idx, customer_type in enumerate(types, start=1):
            base_income: float = float(np.random.uniform(1800, 9000))

            rows.extend(
                self._builder \
                .with_archetype(customer_type) \
                .with_customer_id(f"C{idx:04d}") \
                .with_base_income(base_income) \
                .with_credit_limit(base_income * float(np.random.uniform(2.0, 4.5))) \
                .build()
            )

        df: pd.DataFrame = pd.DataFrame(rows)

        assert len(df) == settings.N_CUSTOMERS * settings.MONTHS

        return df

        
