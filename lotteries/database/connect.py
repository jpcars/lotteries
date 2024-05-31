import numpy as np
import psycopg2
from lotteries.database.config import load_config


def execute_command(command, values=None, has_return_value: bool = False):
    """Connect to the PostgreSQL database server and execute command"""
    config = load_config()
    try:
        with psycopg2.connect(**config) as conn:
            with conn.cursor() as cur:
                cur.execute(command, values)
                if has_return_value:
                    return cur.fetchall()
    except (psycopg2.DatabaseError, Exception) as error:
        print(error)
