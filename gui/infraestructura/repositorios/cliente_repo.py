import sqlite3


def insert_customer(conn: sqlite3.Connection, customer_id: str, archetype: str) -> None:
    sql: str = """
        INSERT INTO customers (customer_id, archetype)
        VALUES (%(customer_id)s, %(archetype)s)
        ;
    """
    try:
        cursor = conn.cursor()
        cursor.execute(sql, {
            'customer_id': customer_id,
            'archetype': archetype
        })

        conn.commit()
    
    except sqlite3.Error:
        pass

    finally:
        conn.close()

def get_customers(conn: sqlite3.Connection) -> list[dict]:
    sql: str = """
        SELECT
            customer_id,
            archetype
        FROM customers
        ;
    """

    try:
        cursor = conn.cursor()
        cursor.execute(sql)
        
        return cursor.fetchall()

    except sqlite3.Error:
        pass

    finally:
        conn.close()

def get_customer(conn: sqlite3.Connection, customer_id: str) -> dict | None:
    sql: str = """
        SELECT
            customer_id,
            archetype
        FROM customers
        WHERE
            customer_id = %(customer_id)s
        ;
    """

    try:
        cursor = conn.cursor()
        cursor.execute(sql, {'customer_id': customer_id})
        
        return cursor.fetchall()

    except sqlite3.Error:
        pass

    finally:
        conn.close()