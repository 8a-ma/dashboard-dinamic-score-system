from pathlib import Path


class Settings:
    SEED: int = 42
    N_CUSTOMERS:int = 500
    MONTHS: int = 24

    PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
    DB_PATH: Path = PROJECT_ROOT / "db"

    RAW_TRANSACTIONS_PATH: Path = DB_PATH / "raw_transactions.csv"
    FEATURES_PATH: Path = DB_PATH / "features_dinamicos.csv"
    LOGISTICS_MODEL_PATH: Path = DB_PATH / "modelo_logistico.pkl"
    LOGISTICS_MODEL_METRICS_PATH: Path = DB_PATH / "metricas_baseline.json"

    MATRIX_SYSTEM_PATH: Path = DB_PATH / "matrices_sistema.npz"


settings = Settings()