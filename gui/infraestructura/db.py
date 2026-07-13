import sqlite3
from pathlib import Path


def inicializar_db(path: Path) -> sqlite3.Connection:
    assert path.parent.exists()
    assert path.suffix == ".db"

    conn: sqlite3.Connection = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row

    conn.execute("""
        CREATE TABLE IF NOT EXISTS customers (
            customer_id TEXT PRIMARY KEY,
            archetype   TEXT NOT NULL
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS monthly_states (
            customer_id       TEXT,
            month             INTEGER,
            income            REAL,
            credit_limit      REAL,
            utilization_rate  REAL,
            outstanding_debt  REAL,
            payment_amount    REAL,
            days_in_default   INTEGER,
            num_transactions  INTEGER,
            transaction_vol   REAL,
            default_indicator INTEGER,
            PRIMARY KEY (customer_id, month)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS estimated_states (
            customer_id    TEXT,
            month          INTEGER,
            x_hat_debt     REAL,
            x_hat_income   REAL,
            x_hat_util     REAL,
            p_trace        REAL,
            score_dinamico REAL,
            PRIMARY KEY (customer_id, month)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS decisions (
            customer_id       TEXT,
            month             INTEGER,
            limit_recomendado REAL,
            score_dinamico    REAL,
            score_logistico   REAL,
            PRIMARY KEY (customer_id, month)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS metrics (
            modelo  TEXT,
            metrica TEXT,
            valor   REAL,
            mes     INTEGER,
            PRIMARY KEY (modelo, metrica, mes)
        )
    """)

    conn.commit()

    assert path.exists()

    return conn


def obtener_conexion(path: Path) -> sqlite3.Connection:
    assert path.parent.exists()
    assert path.suffix == ".db"

    if not path.exists():
        conn = inicializar_db(path)

        return conn

    conn: sqlite3.Connection = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row

    return conn