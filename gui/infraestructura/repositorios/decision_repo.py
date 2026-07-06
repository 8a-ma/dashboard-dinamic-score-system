"""
**`insertar_decision(conn, decision: dict) -> None`**  
**`obtener_decisiones(conn, customer_id: str) -> list[dict]`**  
**`obtener_ultimas_n(conn, customer_id: str, n: int) -> list[dict]`**
"""
import sqlite3


def insert_decision(conn: sqlite3.Connection, decision: dict) -> None:
    cols: str = ", ".join(decision.keys())
    values: str = ", ".join([f":{k}" for k in decision.keys()])

    sql: str = f"INSERT INTO decisions ({cols}) VALUES ({values})"

    try:
        cursor = conn.cursor()
        cursor.execute(sql, decision)
        conn.commit()
    
    except sqlite3.Error:
        pass

    finally:
        conn.close()


def get_decisions(conn: sqlite3.Connection, customer_id: str) -> list[dict]:
    sql: str = """
        SELECT
            *
        FROM decisions
        WHERE
            customer_id = ?
    """

    try:
        cursor = conn.cursor()
        cursor.execute(sql, (customer_id,))
        return cursor.fetchall()
    
    except sqlite3.Error:
        pass

    finally:
        conn.close()
    

def get_n_decisions(conn: sqlite3.Connection, customer_id: str, n: int) -> list[dict]:
    sql: str = """
        SELECT
            *
        FROM decisions
        WHERE
            customer_id = ?
        ORDER BY
            month DESC
        LIMIT ?
    """

    try:
        cursor = conn.cursor()
        cursor.execute(sql, (customer_id, n))
        return cursor.fetchall()
    
    except sqlite3.Error:
        pass

    finally:
        conn.close()