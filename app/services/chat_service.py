from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
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

    chain = (
        graph
        | RunnableLambda(lambda state: state["messages"][-1])
        | parser
    )

    return chain.invoke(
        init_state,
        config=config
    )