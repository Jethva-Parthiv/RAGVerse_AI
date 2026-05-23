from langgraph.graph import StateGraph, START, END
from app.database.postgres import get_postgres_checkpointer
from app.graph.nodes.chat_node import chat_node
from app.graph.state import State


# Initialize once
checkpointer = get_postgres_checkpointer()


def get_graph():
    """
    Create and compile the LangGraph workflow.
    """

    builder = StateGraph(State)

    # Add nodes
    builder.add_node("chat_node", chat_node)

    # Define flow
    builder.add_edge(START, "chat_node")
    builder.add_edge("chat_node", END)

    # Compile graph
    graph = builder.compile(
        checkpointer=checkpointer
    )

    return graph