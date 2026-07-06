import sqlite3
from pathlib import Path
from settings.settings import settings


class DatabaseManager:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        
        self.conn: sqlite3.Connection | None = None
        self.connect()
    
    def connect(self) -> None:
        try:
            self.conn = sqlite3.connect(self.db_path)
        
        except sqlite3.Error as e:
            return
    
    def execute(self, query_sql: str, params: tuple = ()):
        """Insert, Update, Delete, Create"""
        try:
            cursor = self.conn.cursor()
            cursor.execute(query_sql, params)
            self.conn.commit()

            return cursor.rowcount

        except sqlite3.Error as e:
            self.conn.rollback()
            return None
    
    def query(self, query_sql: str, params: tuple = ()):
        "Select"

        try:
            cursor = self.conn.cursor()
            cursor.execute(query_sql, params)
            
            return cursor.fetchall()

        except sqlite3.Error as e:
            return []

    def close(self):
        if self.conn:
            self.conn.close()

db = DatabaseManager(settings.SQLITE_DB)


def init_db():
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS customers (
            customer_id TEXT PRIMARY KEY,
            archetype   TEXT NOT NULL
        );
        """
    )

    db.execute(
        """
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
        );
        """
    )

    db.execute(
        """
        CREATE TABLE IF NOT EXISTS estimated_states (
            customer_id    TEXT,
            month          INTEGER,
            x_hat_debt     REAL,
            x_hat_income   REAL,
            x_hat_util     REAL,
            x_hat_days     REAL,
            p_trace        REAL,
            score_dinamico REAL,
            PRIMARY KEY (customer_id, month)
        );
        """
    )

    db.execute(
        """
        CREATE TABLE IF NOT EXISTS decisions (
            customer_id       TEXT,
            month             INTEGER,
            limit_recomendado REAL,
            score_dinamico    REAL,
            score_logistico   REAL,
            PRIMARY KEY (customer_id, month)
        );
        """
    )

    db.execute(
        """
        CREATE TABLE IF NOT EXISTS metrics (
            modelo TEXT,
            metrica TEXT,
            valor  REAL,
            mes    INTEGER,
            PRIMARY KEY (modelo, metrica, mes)
        );
        """
    )