from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser

from app.graph.builder import get_graph


graph = get_graph()
parser = StrOutputParser()


def get_chat_response(query: str, thread_id: str) -> str:
    init_state = {
        "messages": [HumanMessage(content=query)],
        "query": query
    }

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    response = graph.invoke(
        init_state,
        config=config
    )["messages"][-1]

    return parser.invoke(response)