import logging
import numpy as np
from settings.settings import settings
from modelos.dominio.arquetipos import ARCHETYPE_PARAMS


class CustomerBuilder:
    def __init__(self):
        self._customer_id: str | None = None
        self._archetype: str | None = None
        self._base_income: float | None = None
        self._credit_limit: float | None = None

        self._DISPATCH: dict[str, callable[[int, float], tuple[dict, float]]] = {
            "good": self._build_month_good,
            "recurrent": self._build_month_recurrent,
            "over": self._build_month_over,
            "fraud": self._build_month_fraud,
            "low": self._build_month_low
        }

    def with_customer_id(self, customer_id: str) -> 'CustomerBuilder':
        assert customer_id[0] == "C"

        self._customer_id = customer_id

        return self
    
    def with_archetype(self, archetype: str) -> 'CustomerBuilder':
        assert archetype in ARCHETYPE_PARAMS

        self._archetype = archetype

        return self

    def with_base_income(self, base_income: float) -> 'CustomerBuilder':
        assert base_income >= 0

        self._base_income = base_income

        return self

    def with_credit_limit(self, credit_limit: float) -> 'CustomerBuilder':
        assert credit_limit >= 0

        self._credit_limit = credit_limit

        return self

    def build(self) -> list[dict]:
        assert self._customer_id is not None
        assert self._archetype is not None
        assert self._base_income is not None
        assert self._credit_limit is not None


        drift_start, drift_end, drift_intensity = self._init_drift_params()
        rows: list[dict[str, object]] = []
        previous_row: dict | None = None

        current_limit: float = self._credit_limit
        logging.debug(f'{self.__class__.__name__}.build started - customer_id={self._customer_id} archetype={self._archetype} months={settings.MONTHS}')

        try:
            for month in range(1, settings.MONTHS + 1):
                row, current_limit = self._DISPATCH[self._archetype](month, current_limit)
                row["customer_id"] = self._customer_id
                row["month"] = month

                row = self._apply_autocorrelation(row, previous_row)
                row = self._apply_segment_drift(row, month, drift_start, drift_end, drift_intensity)
                row = self._apply_exogenous_shock(row)

                rows.append(row)
                previous_row = row.copy()
            
            assert len(rows) == settings.MONTHS

            logging.debug(f'{self.__class__.__name__}.build completed - customer_id={self._customer_id} rows={len(rows)}')
            
            return rows
        except AssertionError as e:
            logging.error(f'{self.__class__.__name__}.build failed - customer_id={self._customer_id} reason={e}')
    
    def _clip(self, v: float, a: float, b: float) -> float:
        assert a <= b, f"a={a} debe ser ≤ b={b}"
        result: float = max(a, min(b, float(v)))
        assert a <= result <= b
        return result

    def _build_month_good(self, month: int, current_limit: float) -> tuple[dict[str, object], float]:
        assert 1 <= month <= settings.MONTHS
        assert current_limit > 0.0

        season: float = 1 + 0.03 * np.sin(month / 12 * 2 * np.pi)
        growth: float = 1 + month * float(np.random.uniform(ARCHETYPE_PARAMS[self._archetype]['income_growth_min'], ARCHETYPE_PARAMS[self._archetype]['income_growth_max']))

        income: float = self._base_income * growth * season * float(np.random.normal(1, ARCHETYPE_PARAMS[self._archetype]['income_std']))
        new_limit: float = current_limit * float(np.random.normal(1.002, 0.002))
        utilization: float = self._clip(float(np.random.normal(
                ARCHETYPE_PARAMS[self._archetype]['utilization_mean'],
                ARCHETYPE_PARAMS[self._archetype]['utilization_std']
            )),
            ARCHETYPE_PARAMS[self._archetype]['utilization_min'],
            ARCHETYPE_PARAMS[self._archetype]['utilization_max']
        )
        debt: float = utilization * new_limit
        payment: float = debt * float(np.random.uniform(ARCHETYPE_PARAMS[self._archetype]['payment_factor_min'], ARCHETYPE_PARAMS[self._archetype]['payment_factor_max']))

        row: dict[str, object] = {
            "income": round(income, 2),
            "credit_limit": round(new_limit, 2),
            "utilization_rate": round(utilization, 3),
            "outstanding_debt": round(debt, 2),
            "payment_amount": round(payment, 2),
            "num_transactions": int(np.random.randint(ARCHETYPE_PARAMS[self._archetype]['transactions_min'], ARCHETYPE_PARAMS[self._archetype]['transactions_max'])),
            "transaction_volatility": round(float(np.random.uniform(ARCHETYPE_PARAMS[self._archetype]['transaction_vol_min'], ARCHETYPE_PARAMS[self._archetype]['transaction_vol_max'])), 3),
            "default_indicator": 0,
        }

        return row, new_limit

    def _build_month_recurrent(self, month: int, current_limit: float) -> tuple[dict[str, object], float]:
        assert 1 <= month <= settings.MONTHS
        assert current_limit > 0.0

        season: float = 1 + 0.03 * np.sin(month / 12 * 2 * np.pi)
        shock: float = 1.0

        if np.random.rand() < 0.15:
            shock -= float(np.random.uniform(0.15, 0.35))

        income: float = self._base_income * season * shock * float(np.random.normal(1, ARCHETYPE_PARAMS[self._archetype]['income_std']))
        utilization: float = self._clip(float(np.random.normal(
                ARCHETYPE_PARAMS[self._archetype]['utilization_mean'],
                ARCHETYPE_PARAMS[self._archetype]['utilization_std']
            )),
            ARCHETYPE_PARAMS[self._archetype]['utilization_min'],
            ARCHETYPE_PARAMS[self._archetype]['utilization_max']
        )

        debt: float = utilization * current_limit
        payment: float = debt * float(np.random.uniform(ARCHETYPE_PARAMS[self._archetype]['payment_factor_min'], ARCHETYPE_PARAMS[self._archetype]['payment_factor_max']))
        
        row: dict[str, object] = {
            "income": round(income, 2),
            "credit_limit": round(current_limit, 2),
            "utilization_rate": round(utilization, 3),
            "outstanding_debt": round(debt, 2),
            "payment_amount": round(payment, 2),
            "num_transactions": int(np.random.randint(ARCHETYPE_PARAMS[self._archetype]['transactions_min'], ARCHETYPE_PARAMS[self._archetype]['transactions_max'])),
            "transaction_volatility": round(float(np.random.uniform(ARCHETYPE_PARAMS[self._archetype]['transaction_vol_min'], ARCHETYPE_PARAMS[self._archetype]['transaction_vol_max'])), 3),
            "default_indicator": 0,
        }

        return row, current_limit

    def _build_month_over(self, month: int, current_limit: float) -> tuple[dict[str, object], float]:
        assert 1 <= month <= settings.MONTHS
        assert current_limit > 0.0

        season: float = 1 + 0.03 * np.sin(month / 12 * 2 * np.pi)
        trend: float = 1.0

        if month >= 17:
            trend -= (month - 16) * np.random.uniform(ARCHETYPE_PARAMS[self._archetype]["income_decay_min"], ARCHETYPE_PARAMS[self._archetype]["income_decay_max"])
        
        trend = max(trend, 0.1)

        income: float = self._base_income * trend * season * float(np.random.normal(1, ARCHETYPE_PARAMS[self._archetype]['income_std']))
        utilization: float = self._clip(float(np.random.normal(
                ARCHETYPE_PARAMS[self._archetype]['utilization_mean'],
                ARCHETYPE_PARAMS[self._archetype]['utilization_std']
            )),
            ARCHETYPE_PARAMS[self._archetype]['utilization_min'],
            ARCHETYPE_PARAMS[self._archetype]['utilization_max']
        )

        debt: float = utilization * current_limit
        payment: float = debt * float(np.random.uniform(ARCHETYPE_PARAMS[self._archetype]['payment_factor_min'], ARCHETYPE_PARAMS[self._archetype]['payment_factor_max']))

        row: dict[str, object] = {
            "income": round(income, 2),
            "credit_limit": round(current_limit, 2),
            "utilization_rate": round(utilization, 3),
            "outstanding_debt": round(debt, 2),
            "payment_amount": round(payment, 2),
            "num_transactions": int(np.random.randint(ARCHETYPE_PARAMS[self._archetype]['transactions_min'], ARCHETYPE_PARAMS[self._archetype]['transactions_max'])),
            "transaction_volatility": round(float(np.random.uniform(ARCHETYPE_PARAMS[self._archetype]['transaction_vol_min'], ARCHETYPE_PARAMS[self._archetype]['transaction_vol_max'])), 3),
            "default_indicator": 1 if month >= 18 else 0,
        }

        return row, current_limit

    def _build_month_fraud(self, month: int, current_limit: float) -> tuple[dict[str, object], float]:
        assert 1 <= month <= settings.MONTHS
        assert current_limit > 0.0

        income: float = self._base_income * float(np.random.uniform(ARCHETYPE_PARAMS[self._archetype]['income_factor_min'], ARCHETYPE_PARAMS[self._archetype]['income_factor_max']))

        if month <= 15:
            utilization: float = self._clip(float(np.random.normal(
                    ARCHETYPE_PARAMS[self._archetype]['utilization_mean'],
                    ARCHETYPE_PARAMS[self._archetype]['utilization_std']
                )),
                ARCHETYPE_PARAMS[self._archetype]['utilization_min'],
                ARCHETYPE_PARAMS[self._archetype]['utilization_max']
            )
            debt: float = utilization * current_limit
            payment: float = debt * float(np.random.uniform(ARCHETYPE_PARAMS[self._archetype]['payment_factor_min'], ARCHETYPE_PARAMS[self._archetype]['payment_factor_max']))
            return {
                "income": round(income, 2),
                "credit_limit": round(current_limit, 2),
                "utilization_rate": round(utilization, 3),
                "outstanding_debt": round(debt, 2),
                "payment_amount": round(payment, 2),
                "num_transactions": int(np.random.randint(ARCHETYPE_PARAMS[self._archetype]['transactions_min'], ARCHETYPE_PARAMS[self._archetype]['transactions_max'])),
                "transaction_volatility": round(float(np.random.uniform(ARCHETYPE_PARAMS[self._archetype]['transaction_vol_min'], ARCHETYPE_PARAMS[self._archetype]['transaction_vol_max'])), 3),
                "default_indicator": 0,
            }, current_limit
        
        debt = current_limit

        return {
            "income": round(income, 2),
            "credit_limit": round(current_limit, 2),
            "utilization_rate": 1.0,
            "outstanding_debt": round(debt, 2),
            "payment_amount": 0.0,
            "num_transactions": 0,
            "transaction_volatility": 0.0,
            "default_indicator": 1,
        }, current_limit


    def _build_month_low(self, month: int, current_limit: float) -> tuple[dict[str, object], float]:
        assert 1 <= month <= settings.MONTHS
        assert current_limit > 0.0

        active: bool = bool(np.random.rand() < 0.35)

        if active:
            income: float = self._base_income * float(np.random.normal(1, ARCHETYPE_PARAMS[self._archetype]['income_std']))
            utilization: float = np.random.uniform(ARCHETYPE_PARAMS[self._archetype]["utilization_min"], ARCHETYPE_PARAMS[self._archetype]["utilization_max"])
            debt: float = utilization * current_limit
            payment: float = debt * float(np.random.uniform(ARCHETYPE_PARAMS[self._archetype]['payment_factor_min'], ARCHETYPE_PARAMS[self._archetype]['payment_factor_max']))

            return {
                "income": round(income, 2),
                "credit_limit": round(current_limit, 2),
                "utilization_rate": round(utilization, 3),
                "outstanding_debt": round(debt, 2),
                "payment_amount": round(payment, 2),
                "days_in_default": 0,
                "num_transactions": int(np.random.randint(ARCHETYPE_PARAMS[self._archetype]['transactions_min'], ARCHETYPE_PARAMS[self._archetype]['transactions_max'])),
                "transaction_volatility": round(float(np.random.uniform(ARCHETYPE_PARAMS[self._archetype]['transaction_vol_min'], ARCHETYPE_PARAMS[self._archetype]['transaction_vol_max'])), 3),
                "default_indicator": 0,
            }, current_limit

        income_inactivo: float = np.nan if np.random.rand() < 0.6 else self._base_income

        return {
            "income": income_inactivo,
            "credit_limit": round(current_limit, 2),
            "utilization_rate": 0.0,
            "outstanding_debt": 0.0,
            "payment_amount": 0.0,
            "days_in_default": 0,
            "num_transactions": 0,
            "transaction_volatility": np.nan,
            "default_indicator": 0,
        }, current_limit
    
    def _apply_exogenous_shock(self, row: dict) -> dict:
        assert "outstanding_debt" in row
        assert "income" in row
        
        if np.random.rand() >= ARCHETYPE_PARAMS[self._archetype]['shock_prob']:
            return row
        
        if row["outstanding_debt"] > 0:
            windfall: float = np.random.uniform(0.1, 0.5) * row["outstanding_debt"]
            row["outstanding_debt"] = max(0.0, row["outstanding_debt"] - windfall)

        if row["income"] is not None and not np.isnan(row["income"]):
            row["income"] = row["income"] * np.random.uniform(1.05, 1.20)

        return row

    def _init_drift_params(self) -> tuple[int, int, float]:
        drift_start: int = int(np.random.randint(4, 18))
        drift_duration: int = int(np.random.randint(3, 7))
        drift_end: int = min(drift_start + drift_duration, settings.MONTHS)
        drift_intensity: float = float(np.random.uniform(0.05, 0.25))

        assert 1 <= drift_start < drift_end <= settings.MONTHS
        assert 0.0 < drift_intensity <= 0.25

        return drift_start, drift_end, drift_intensity
    
    def _apply_autocorrelation(self, row: dict, previous_row: dict | None, rho: float = 0.25) -> dict:
        assert 0.0 <= rho <= 1.0
        assert "outstanding_debt" in row

        if previous_row is None: return row

        row["outstanding_debt"] = rho * previous_row["outstanding_debt"] + (1 - rho) * row['outstanding_debt']

        prev_debt: float = max(previous_row['outstanding_debt'], 1.0)
        payment_ratio: float = previous_row['payment_amount'] / prev_debt

        if payment_ratio > 0.90:
            reduction: float = np.random.uniform(0.01, 0.05) * row['outstanding_debt']
            row['outstanding_debt'] = max(0.0, row['outstanding_debt'] - reduction)
        
        elif payment_ratio < 0.30:
            penalty: float = np.random.uniform(0.01, 0.04) * row['outstanding_debt']
            row['outstanding_debt'] = row['outstanding_debt'] + penalty
        
        if row['credit_limit'] > 0.0:
            row['utilization_rate'] = self._clip(row['outstanding_debt'] / row['credit_limit'], 0.0, 1.0)

        assert row['outstanding_debt'] >= 0.0
        assert 0.0 <= row['utilization_rate'] <= 1.0

        return row

    def _apply_segment_drift(self, row: dict, month: int, drift_start: int, drift_end: int, drift_intensity: float) -> dict:
        assert drift_start < drift_end
        assert 0.0 <= drift_intensity <= 1.0
        assert "outstanding_debt" in row and "payment_amount" in row

        if not (drift_start <= month <= drift_end): return row

        window_size: int = max(drift_end - drift_start, 1)
        progress: float = (month - drift_start) / window_size
        alpha: float = drift_intensity * float(np.sin(np.pi * progress))

        row['outstanding_debt'] = row["outstanding_debt"] * (1.0 + alpha * float(np.random.uniform(0.03, 0.12)))
        row['payment_amount'] = max(0.0, row["payment_amount"] * (1.0 + alpha * float(np.random.uniform(0.05, 0.20))))

        if row['credit_limit'] > 0.0:
            row['utilization_rate'] = self._clip(row['outstanding_debt'] / row['credit_limit'], 0.0, 1.0)

        assert row['outstanding_debt'] >= 0.0
        assert row['payment_amount'] >= 0.0

        return row