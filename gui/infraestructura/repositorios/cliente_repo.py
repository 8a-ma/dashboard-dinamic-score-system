import sqlite3


def insert_customer(conn: sqlite3.Connection, customer_id: str, archetype: str) -> None:
    assert customer_id.strip() != ""
    assert archetype in {"good", "recurrent", "over", "fraud", "low"}

    sql: str = """
        INSERT OR IGNORE INTO customers (customer_id, archetype)
        VALUES (:customer_id, :archetype)
    """
    cursor: sqlite3.Cursor = conn.cursor()
    cursor.execute(sql, {"customer_id": customer_id, "archetype": archetype})
    conn.commit()


def get_customers(conn: sqlite3.Connection) -> list[dict]:
    sql: str = """
        SELECT customer_id, archetype
        FROM customers
        ORDER BY customer_id
    """

    cursor: sqlite3.Cursor = conn.cursor()
    cursor.execute(sql)
    rows: list[sqlite3.Row] = cursor.fetchall()
 
    return [dict(row) for row in rows]


def get_customer(conn: sqlite3.Connection, customer_id: str) -> dict | None:
    assert customer_id.strip() != ""

    sql: str = """
        SELECT customer_id, archetype
        FROM customers
        WHERE customer_id = :customer_id
    """

    cursor: sqlite3.Cursor = conn.cursor()
    cursor.execute(sql, {"customer_id": customer_id})
    row: sqlite3.Row | None = cursor.fetchone()
 
    return dict(row) if row is not None else None