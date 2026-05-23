from psycopg import connect
from psycopg.errors import OperationalError
from langgraph.checkpoint.postgres import PostgresSaver

from app.core.settings import DATABASE_URL


def get_postgres_checkpointer() -> PostgresSaver:
    """
    Initialize PostgreSQL-based LangGraph checkpointer.
    """

    try:
        conn = connect(
            conninfo=DATABASE_URL,
            autocommit=True,
        )

        checkpointer = PostgresSaver(conn)
        checkpointer.setup()

        return checkpointer

    except OperationalError as error:
        raise RuntimeError(
            "Failed to connect to PostgreSQL database."
        ) from error