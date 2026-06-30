import os
import sys
import argparse
import numpy as np
import pandas as pd
from typing import Any
from pathlib import Path
from settings.settings import settings


# SEED: int = 42
# N_CUSTOMERS:int = 500
# MONTHS: int = 24

# PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
# OUTPUT_DIR: Path = PROJECT_ROOT / "db"
OUTPUT_FILE: Path = settings.OUTPUT_DIR / "raw_transactions.csv"


def _clip(v: float, a: float, b: float) -> float:
    assert a <= b, f"a={a} debe ser ≤ b={b}"
    result: float = max(a, min(b, float(v)))
    assert a <= result <= b
    return result


def parsear_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Genera dataset sintético de transacciones crediticias"
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerar CSV aunque ya exista"
    )

    args: argparse.Namespace = parser.parse_args()

    assert isinstance(args, argparse.Namespace)
    assert isinstance(args.force, bool)

    return args


def calulate_composition(n: int) -> dict[str, int]:
    assert n > 0
    assert n == settings.N_CUSTOMERS

    comp: dict[str, int] = {
        "good": int(0.40 * n),
        "recurrent": int(0.25 * n),
        "over": int(0.20 * n),
        "fraud": int(0.05 * n),
    }

    comp["low"] = n - sum(comp.values())

    assert sum(comp.values()) == n

    return comp


def generate_customer_type(composition: dict[str, int]) -> list[str]:
    assert len(composition) == 5
    assert sum(composition.values()) == settings.N_CUSTOMERS

    types: list[str] = []

    for type, quant in composition.items():
        types += [type] * quant
    
    np.random.shuffle(types)

    assert len(types) == settings.N_CUSTOMERS

    return types


def _row_good(month: int, base_income: float, credit_limit: float) -> tuple[dict[str, Any], float]:
    assert 1 <= month <= settings.MONTHS
    assert credit_limit > 0

    season: float = 1 + 0.03 * np.sin(month / 12 * 2 * np.pi)
    growth: float = 1 + month * float(np.random.uniform(0.002, 0.008))
    income: float = base_income * growth * season * float(np.random.normal(1, 0.03))
    new_limit: float = credit_limit * float(np.random.normal(1.002, 0.002))
    utilization: float = _clip(float(np.random.normal(0.45, 0.07)), 0.30, 0.60)
    debt: float = utilization * new_limit
    payment: float = debt * float(np.random.uniform(0.92, 1.05))

    row: dict[str, Any] = {
        "income": round(income, 2),
        "credit_limit": round(new_limit, 2),
        "utilization_rate": round(utilization, 3),
        "outstanding_debt": round(debt, 2),
        "payment_amount": round(payment, 2),
        "days_in_default": 0,
        "num_transactions": int(np.random.randint(8, 16)),
        "transaction_volatility": round(float(np.random.uniform(0.05, 0.15)), 3),
        "default_indicator": 0,
    }

    return row, new_limit


def _row_recurrent(month: int, base_income: float, credit_limit: float) -> dict[str, Any]:
    assert 1 <= month <= settings.MONTHS
    assert credit_limit > 0

    season: float = 1 + 0.03 * np.sin(month / 12 * 2 * np.pi)
    shock: float = 1.0

    if np.random.rand() < 0.15:
        shock -= float(np.random.uniform(0.15, 0.35))

    income: float = base_income * season * shock * float(np.random.normal(1, 0.08))
    utilization: float = _clip(float(np.random.normal(0.72, 0.08)), 0.60, 0.85)
    debt: float = utilization * credit_limit
    payment: float = debt * float(np.random.uniform(0.55, 0.95))
    default_days: int = 0

    if np.random.rand() < 0.25:
        default_days = int(np.random.randint(10, 31))
    
    return {
        "income": round(income, 2),
        "credit_limit": round(credit_limit, 2),
        "utilization_rate": round(utilization, 3),
        "outstanding_debt": round(debt, 2),
        "payment_amount": round(payment, 2),
        "days_in_default": default_days,
        "num_transactions": int(np.random.randint(4, 13)),
        "transaction_volatility": round(float(np.random.uniform(0.2, 0.4)), 3),
        "default_indicator": 0,
    }


def _row_over(month: int, base_income: float, credit_limit: float) -> dict[str, Any]:
    assert 1 <= month <= settings.MONTHS
    assert credit_limit > 0

    season: float = 1 + 0.03 * np.sin(month / 12 * 2 * np.pi)
    trend: float = 1.0

    if month >= 17:
        trend = 1 - (month - 16) * float(np.random.uniform(0.04, 0.07))

    income: float = base_income * trend * season * float(np.random.normal(1, 0.05))
    utilization: float = _clip(float(np.random.normal(0.96, 0.02)), 0.90, 1.00)
    debt: float = utilization * credit_limit
    payment: float = debt * float(np.random.uniform(0.02, 0.08))

    if month < 12:
        default_days: int = 0
    
    elif month < 18:
        default_days = (month - 11) * int(np.random.randint(3, 6))
    
    else:
        default_days = (month - 17) * int(np.random.randint(12, 20)) + 35

    transactions: int = (
        int(np.random.randint(5, 9)) if month < 17 else int(np.random.randint(2, 6))
    )

    return {
        "income": round(income, 2),
        "credit_limit": round(credit_limit, 2),
        "utilization_rate": round(utilization, 3),
        "outstanding_debt": round(debt, 2),
        "payment_amount": round(payment, 2),
        "days_in_default": default_days,
        "num_transactions": transactions,
        "transaction_volatility": round(float(np.random.uniform(0.4, 0.7)), 3),
        "default_indicator": 1 if month >= 18 else 0,
    }


def _row_fraud(month: int, base_income: float, credit_limit: float) -> dict[str, Any]:
    assert 1 <= month <= settings.MONTHS
    assert credit_limit > 0

    income: float = base_income * float(np.random.uniform(1.8, 2.5))

    if month <= 15:
        utilization: float = _clip(float(np.random.normal(0.75, 0.05)), 0.65, 0.85)
        debt: float = utilization * credit_limit
        payment: float = debt * float(np.random.uniform(0.85, 1.0))
        return {
            "income": round(income, 2),
            "credit_limit": round(credit_limit, 2),
            "utilization_rate": round(utilization, 3),
            "outstanding_debt": round(debt, 2),
            "payment_amount": round(payment, 2),
            "days_in_default": 0,
            "num_transactions": int(np.random.randint(12, 21)),
            "transaction_volatility": round(float(np.random.uniform(0.05, 0.10)), 3),
            "default_indicator": 0,
        }

    debt = credit_limit

    return {
        "income": round(income, 2),
        "credit_limit": round(credit_limit, 2),
        "utilization_rate": 1.0,
        "outstanding_debt": round(debt, 2),
        "payment_amount": 0.0,
        "days_in_default": 60 + (month - 16) * 10,
        "num_transactions": 0,
        "transaction_volatility": 0.0,
        "default_indicator": 1,
    }


def _row_low(month: int, base_income: float, credit_limit: float) -> dict[str, Any]:
    assert 1 <= month <= settings.MONTHS
    assert credit_limit > 0

    active: bool = bool(np.random.rand() < 0.35)

    if active:
        income: float = base_income * float(np.random.normal(1, 0.04))
        utilization: float = float(np.random.uniform(0.05, 0.30))
        debt: float = utilization * credit_limit
        payment: float = debt * float(np.random.uniform(0.8, 1.1))

        return {
            "income": round(income, 2),
            "credit_limit": round(credit_limit, 2),
            "utilization_rate": round(utilization, 3),
            "outstanding_debt": round(debt, 2),
            "payment_amount": round(payment, 2),
            "days_in_default": 0,
            "num_transactions": int(np.random.randint(1, 4)),
            "transaction_volatility": round(float(np.random.uniform(0.01, 0.05)), 3),
            "default_indicator": 0,
        }

    income_inactivo: float = np.nan if np.random.rand() < 0.6 else base_income

    return {
        "income": income_inactivo,
        "credit_limit": round(credit_limit, 2),
        "utilization_rate": 0.0,
        "outstanding_debt": 0.0,
        "payment_amount": 0.0,
        "days_in_default": 0,
        "num_transactions": 0,
        "transaction_volatility": np.nan,
        "default_indicator": 0,
    }


_DISPATCH: dict[str, Any] = {
    "recurrent": _row_recurrent,
    "over": _row_over,
    "fraud": _row_fraud,
    "low": _row_low,
}


def generate_customer(customer_id: str, customer_type: str, base_income: float, credit_limit: float) -> list[dict[str, Any]]:
    assert customer_type in {"good", "recurrent", "over", "fraud", "low"}
    assert base_income > 0 and credit_limit > 0

    rows: list[dict[str, Any]] = []
    current_limit: float = credit_limit

    for month in range(1, settings.MONTHS + 1):
        if customer_type == "good":
            row, current_limit = _row_good(month, base_income, current_limit)

        else:
            row = _DISPATCH[customer_type](month, base_income, current_limit)

        row["customer_id"] = customer_id
        row["month"] = month
        rows.append(row)

    assert len(rows) == settings.MONTHS

    return rows


def generate_dataset(types: list[str]) -> pd.DataFrame:
    assert len(types) == settings.N_CUSTOMERS
    assert set(types) <= {"good", "recurrent", "over", "fraud", "low"}
    
    rows: list[dict[str, Any]] = []

    for idx, customer_type in enumerate(types, start=1):
        customer_id: str = f"C{idx:04d}"
        base_income: float = float(np.random.uniform(1800, 9000))
        credit_limit: float = base_income * float(np.random.uniform(2.0, 4.5))
        rows.extend(generate_customer(customer_id, customer_type, base_income, credit_limit))

    df: pd.DataFrame = pd.DataFrame(rows)

    assert len(df) == settings.N_CUSTOMERS * settings.MONTHS
    assert set(df["customer_id"].str[0]) == {"C"}

    return df


def save_csv(df: pd.DataFrame, path: Path) -> None:
    assert len(df) > 0
    assert path.parent.exists()

    col_order: list[str] = [
        "customer_id", "month", "income", "credit_limit",
        "utilization_rate", "outstanding_debt", "payment_amount",
        "days_in_default", "num_transactions", "transaction_volatility",
        "default_indicator",
    ]

    df[col_order].to_csv(path, index=False, encoding="utf-8")

    assert path.exists()


def main() -> None:
    assert settings.SEED == 42
    np.random.seed(settings.SEED)
    args: argparse.Namespace = parsear_args()

    assert settings.OUTPUT_DIR.parent.exists()
    os.makedirs(settings.OUTPUT_DIR, exist_ok=True)

    if OUTPUT_FILE.exists() and not args.force:
        print(f"Ya existe {OUTPUT_FILE}. Usa --force para regenerar.")
        sys.exit(0)

    composicion: dict[str, int] = calulate_composition(settings.N_CUSTOMERS)
    tipos: list[str] = generate_customer_type(composicion)
    df: pd.DataFrame = generate_dataset(tipos)

    save_csv(df, OUTPUT_FILE)

    print(f"Dataset generado: {OUTPUT_FILE}")
    print(df.head())


if __name__ == "__main__":
    main()


# ============================================================
# Ejemplos de clientes (comentarios)
# ============================================================

# GOOD:
# ingreso estable y creciente
# utilización 30-60%
# sin mora
# pagos completos
#
# RECURRENT:
# ingresos variables
# mora temporal (10-30 días)
# recuperación posterior
#
# OVER-INDEBTED:
# ingresos caen últimos meses
# utilización cercana a 100%
# pagos mínimos
# mora creciente
# default meses 18-24
#
# FRAUD:
# ingresos altos declarados
# pagos normales hasta mes 15
# utilización 100%
# pago = 0 desde mes 16
# default abrupto
#
# LOW INTERACTION:
# meses sin actividad
# pocas transacciones
# ingresos faltantes en algunos meses
# nunca entra en mora