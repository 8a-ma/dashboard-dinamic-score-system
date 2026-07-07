import sqlite3


def insert_decision(conn: sqlite3.Connection, decision: dict) -> None:
    assert "customer_id" in decision
    assert "month" in decision

    cols: str = ", ".join(decision.keys())
    placeholders: str = ", ".join(f":{k}" for k in decision.keys())

    sql: str = f"INSERT OR REPLACE INTO decisions ({cols}) VALUES ({placeholders})"

    cursor: sqlite3.Cursor = conn.cursor()
    cursor.execute(sql, decision)
    conn.commit()


def get_decisions(conn: sqlite3.Connection, customer_id: str) -> list[dict]:
    assert len(customer_id) > 0
    
    sql: str = """
        SELECT *
        FROM decisions
        WHERE customer_id = ?
        ORDER BY month ASC
    """

    cursor: sqlite3.Cursor = conn.cursor()
    cursor.execute(sql, (customer_id,))
    rows: list[sqlite3.Row] = cursor.fetchall()
 
    return [dict(row) for row in rows]
    

def get_last_n_decisions(conn: sqlite3.Connection, customer_id: str, n: int) -> list[dict]:
    assert len(customer_id) > 0
    assert n > 0

    sql: str = """
        SELECT *
        FROM decisions
        WHERE customer_id = ?
        ORDER BY month DESC
        LIMIT ?
    """

    cursor: sqlite3.Cursor = conn.cursor()
    cursor.execute(sql, (customer_id, n))
    rows: list[sqlite3.Row] = cursor.fetchall()
 
    return [dict(row) for row in rows]