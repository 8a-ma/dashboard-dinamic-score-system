import sqlite3


def insert_monthly_state(conn: sqlite3.Connection, state: dict) -> None:
    assert "customer_id" in state
    assert "month" in state

    cols: str = ", ".join(state.keys())
    values: str = ", ".join([f":{k}" for k in state.keys()])

    sql: str = f"INSERT OR REPLACE INTO monthly_states ({cols}) VALUES ({values})"

    cursor: sqlite3.Cursor = conn.cursor()
    cursor.execute(sql, state)
    conn.commit()


def insert_estimate_state(conn: sqlite3.Connection, estimate: dict) -> None:
    assert "customer_id" in estimate
    assert "month" in estimate

    cols: str = ", ".join(estimate.keys())
    values: str = ", ".join([f":{k}" for k in estimate.keys()])

    sql: str = f"INSERT OR REPLACE INTO estimated_states ({cols}) VALUES ({values})"

    cursor: sqlite3.Cursor = conn.cursor()
    cursor.execute(sql, estimate)
    conn.commit()


def get_trajectory(conn: sqlite3.Connection, customer_id: str) -> list[dict]:
    assert isinstance(customer_id, str)
    assert len(customer_id) > 0
    
    sql: str = """
        SELECT *
        FROM monthly_states
        WHERE customer_id = ?
        ORDER BY month ASC
    """

    cursor: sqlite3.Cursor = conn.cursor()
    cursor.execute(sql, (customer_id,))
    rows: list = cursor.fetchall()

    return [dict(row) for row in rows]


def get_estimated_state(conn: sqlite3.Connection, customer_id: str) -> list[dict]:
    assert isinstance(customer_id, str)
    assert len(customer_id) > 0

    sql: str = """
        SELECT *
        FROM estimated_states
        WHERE customer_id = ?
        ORDER BY month ASC
    """

    cursor: sqlite3.Cursor = conn.cursor()
    cursor.execute(sql, (customer_id,))
    rows: list = cursor.fetchall()

    return [dict(row) for row in rows]