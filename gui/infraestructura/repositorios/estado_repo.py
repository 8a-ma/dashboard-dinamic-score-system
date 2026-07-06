import sqlite3


def insert_monthly_state(conn: sqlite3.Connection, state: dict) -> None:
    assert ("customer_id", "month") in list(state.keys())

    cols: str = ", ".join(state.keys())
    values: str = ", ".join([f":{k}" for k in state.keys()])

    sql: str = f"INSERT INTO monthly_states ({cols}) VALUES ({values})"

    try:
        cursor = conn.cursor()
        cursor.execute(sql, state)
        conn.commit()
    
    except sqlite3.Error:
        pass

    finally:
        conn.close()


def insert_estimate_state(conn: sqlite3.Connection, estimate: dict) -> None:
    assert ("customer_id", "month") in list(estimate.keys())

    cols: str = ", ".join(estimate.keys())
    values: str = ", ".join([f":{k}" for k in estimate.keys()])

    sql: str = f"INSERT INTO estimated_states ({cols}) VALUES ({values})"

    try:
        cursor = conn.cursor()
        cursor.execute(sql, estimate)
        conn.commit()
    
    except sqlite3.Error:
        pass

    finally:
        conn.close()

def get_track(conn: sqlite3.Connection, customer_id: str) -> list[dict]:
    sql: str = """
        SELECT
            *
        FROM monthly_states
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


def get_estimate_state(conn: sqlite3.Connection, customer_id: str) -> list[dict]:
    sql: str = """
        SELECT
            *
        FROM estimated_states
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