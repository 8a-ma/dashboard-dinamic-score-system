from pathlib import Path

SEED: int = 42
N_CUSTOMERS:int = 500
MONTHS: int = 24

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
OUTPUT_DIR: Path = PROJECT_ROOT / "db"

class Settings:
    SEED: int = 42
    N_CUSTOMERS:int = 500
    MONTHS: int = 24

    PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
    OUTPUT_DIR: Path = PROJECT_ROOT / "db"


settings = Settings()