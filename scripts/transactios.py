import os
import numpy as np
import pandas as pd
from pathlib import Path

# ============================================================
# Synthetic Credit Dataset Generator
# ============================================================

SEED = 42
np.random.seed(SEED)

N_CUSTOMERS = 500
MONTHS = 24

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "db"
OUTPUT_FILE = OUTPUT_DIR / "raw_transactions.csv"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------------------------------------
# Customer composition
# ------------------------------------------------------------

composition = {
    "good": int(0.40 * N_CUSTOMERS),          # 200
    "recurrent": int(0.25 * N_CUSTOMERS),     # 125
    "over": int(0.20 * N_CUSTOMERS),          # 100
    "fraud": int(0.05 * N_CUSTOMERS),         # 25
}

composition["low"] = (
    N_CUSTOMERS
    - composition["good"]
    - composition["recurrent"]
    - composition["over"]
    - composition["fraud"]
)

# ------------------------------------------------------------

def clip(v, a, b):
    return max(a, min(b, v))

rows = []

customer_types = []

customer_types += ["good"] * composition["good"]
customer_types += ["recurrent"] * composition["recurrent"]
customer_types += ["over"] * composition["over"]
customer_types += ["fraud"] * composition["fraud"]
customer_types += ["low"] * composition["low"]

np.random.shuffle(customer_types)

# ------------------------------------------------------------

for customer_idx, customer_type in enumerate(customer_types, start=1):

    customer_id = f"C{customer_idx:04d}"

    base_income = np.random.uniform(1800, 9000)
    credit_limit = base_income * np.random.uniform(2.0, 4.5)

    debt = 0

    recurring_default = False

    for month in range(1, MONTHS + 1):

        season = 1 + 0.03 * np.sin(month / 12 * 2 * np.pi)

        ##############################################
        # GOOD
        ##############################################

        if customer_type == "good":

            growth = 1 + month * np.random.uniform(0.002, 0.008)

            income = (
                base_income
                * growth
                * season
                * np.random.normal(1, 0.03)
            )

            utilization = clip(np.random.normal(0.45, 0.07), 0.30, 0.60)

            credit_limit *= np.random.normal(1.002, 0.002)

            debt = utilization * credit_limit

            payment = debt * np.random.uniform(0.92, 1.05)

            default_days = 0

            transactions = np.random.randint(8, 16)

            volatility = np.random.uniform(0.05, 0.15)

            default = 0

        ##############################################
        # RECURRENT
        ##############################################

        elif customer_type == "recurrent":

            shock = 1

            if np.random.rand() < 0.15:
                shock -= np.random.uniform(0.15, 0.35)

            income = (
                base_income
                * season
                * shock
                * np.random.normal(1, 0.08)
            )

            utilization = clip(np.random.normal(0.72, 0.08), 0.60, 0.85)

            debt = utilization * credit_limit

            payment = debt * np.random.uniform(0.55, 0.95)

            if np.random.rand() < 0.25:

                recurring_default = True

                default_days = np.random.randint(10, 31)

            else:

                recurring_default = False

                default_days = 0

            transactions = np.random.randint(4, 13)

            volatility = np.random.uniform(0.2, 0.4)

            default = 0

        ##############################################
        # OVER-INDEBTED
        ##############################################

        elif customer_type == "over":

            if month < 17:

                trend = 1

            else:

                trend = 1 - (month - 16) * np.random.uniform(0.04, 0.07)

            income = (
                base_income
                * trend
                * season
                * np.random.normal(1, 0.05)
            )

            utilization = clip(np.random.normal(0.96, 0.02), 0.90, 1.00)

            debt = utilization * credit_limit

            payment = debt * np.random.uniform(0.02, 0.08)

            if month < 12:

                default_days = 0

            elif month < 18:

                default_days = (month - 11) * np.random.randint(3, 6)

            else:

                default_days = (month - 17) * np.random.randint(12, 20) + 35

            volatility = np.random.uniform(0.4, 0.7)

            if month < 17:

                transactions = np.random.randint(5, 9)

            else:

                transactions = np.random.randint(2, 6)

            default = 1 if month >= 18 else 0

        ##############################################
        # FRAUD
        ##############################################

        elif customer_type == "fraud":

            income = base_income * np.random.uniform(1.8, 2.5)

            if month <= 15:

                utilization = clip(
                    np.random.normal(0.75, 0.05),
                    0.65,
                    0.85,
                )

                payment = utilization * credit_limit * np.random.uniform(
                    0.85,
                    1.0,
                )

                default_days = 0

                volatility = np.random.uniform(0.05, 0.10)

                transactions = np.random.randint(12, 21)

                default = 0

            else:

                utilization = 1.0

                payment = 0

                default_days = 60 + (month - 16) * 10

                volatility = 0

                transactions = 0

                default = 1

            debt = utilization * credit_limit

        ##############################################
        # LOW INTERACTION
        ##############################################

        else:

            active = np.random.rand() < 0.35

            if active:

                income = base_income * np.random.normal(1, 0.04)

                utilization = np.random.uniform(0.05, 0.30)

                debt = utilization * credit_limit

                payment = debt * np.random.uniform(0.8, 1.1)

                transactions = np.random.randint(1, 4)

                volatility = np.random.uniform(0.01, 0.05)

            else:

                income = np.nan if np.random.rand() < 0.6 else base_income

                utilization = 0

                debt = 0

                payment = 0

                transactions = 0

                volatility = np.nan

            default_days = 0

            default = 0

        rows.append({

            "customer_id": customer_id,
            "month": month,
            "income": round(income,2) if not np.isnan(income) else np.nan,
            "credit_limit": round(credit_limit,2),
            "utilization_rate": round(utilization,3),
            "outstanding_debt": round(debt,2),
            "payment_amount": round(payment,2),
            "days_in_default": int(default_days),
            "num_transactions": int(transactions),
            "transaction_volatility": (
                round(volatility,3)
                if not np.isnan(volatility)
                else np.nan
            ),
            "default_indicator": int(default)

        })

# ------------------------------------------------------------

df = pd.DataFrame(rows)

df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8"
)

print(f"Dataset generado: {OUTPUT_FILE}")
print(df.head())

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