from pathlib import Path


class Settings:
    SEED: int = 42
    N_CUSTOMERS:int = 500
    MONTHS: int = 24
    TEST_MONTHS: int = 6

    PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
    DB_PATH: Path = PROJECT_ROOT / "db"

    RAW_TRANSACTIONS_PATH: Path = DB_PATH / "raw_transactions.csv"
    FEATURES_PATH: Path = DB_PATH / "features_dinamicos.csv"
    LOGISTICS_MODEL_PATH: Path = DB_PATH / "modelo_logistico.pkl"
    LOGISTICS_MODEL_METRICS_PATH: Path = DB_PATH / "metricas_baseline.json"
    MATRIX_SYSTEM_PATH: Path = DB_PATH / "matrices_sistema.npz"
    COMPARISON_PATH: Path = DB_PATH / "comparacion_modelos.json"

    SQLITE_DB: Path = DB_PATH / "credito.db"

    GUI: Path = PROJECT_ROOT / "gui"
    CSS_GLOBAL: Path = GUI / "global.css"
    CSS_CUSTOMER: Path = GUI / "cliente" / "styles.css"
    CSS_BANK: Path = GUI / "banco" / "styles.css"

    STATES: list[str] = ['outstanding_debt', 'income', 'utilization_rate']
    N_STATES: int = len(STATES)

    CONTROL: list[str] = ['credit_limit']
    N_CONTROL: int = len(CONTROL)

    OBSERVATIONS: list[str] = ['num_transactions', 'payment_amount']
    N_OBSERVATIONS: int = len(OBSERVATIONS)

    RECOVERY_RATE: float = 0.30
    MONTHLY_STATE_COLS: list[str] = [
        'customer_id', 
        'month', 
        'income', 
        'credit_limit', 
        'utilization_rate',
        'outstanding_debt', 
        'payment_amount',
        'num_transactions', 
        'transaction_volatility', 
        'default_indicator'
    ]

    FEATURE_COLUMNS: list[str] = [
        'ratio_deuda_ingreso_ma',
        'tendencia_utilizacion',
        'volatilidad_pagos',
        'utilization_rate',
        'num_transactions'
    ]
    TARGET_COLUMN: str = 'default_indicator'


settings = Settings()