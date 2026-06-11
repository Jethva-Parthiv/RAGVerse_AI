from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from app.graph.state import State
from app.llm.models import get_gemini_chat_model
from app.llm.prompts import BASE_RAG_RULES_HOTPOTQA_DATASET

from app.retrieval.advanced_retriever import get_advanced_retriever

_retriever = None
_llm       = None
_chain     = None


def _get_chain():
    global _retriever, _llm, _chain
    if _chain is None:
        _retriever = get_advanced_retriever()
        _llm       = get_gemini_chat_model()

        prompt = ChatPromptTemplate.from_messages([
            ("system", BASE_RAG_RULES_HOTPOTQA_DATASET),
            MessagesPlaceholder(variable_name="history"),
            ("human", "Context:\n{context}\n\nQuestion: {query}"),
        ])
        _chain = prompt | _llm
    return _retriever, _chain


def chat_node(state: State) -> dict:
    query    = state["query"]
    history  = state["messages"][:-1]   # all but the current human message

    retriever, chain = _get_chain()

    # Advanced retrieval (multi-query + hybrid + re-rank)
    docs    = retriever.retrieve(query)
    context = "\n\n".join(doc.page_content for doc in docs)

    # Generate
    response = chain.invoke({
        "history": history,
        "context": context,
        "query":   query,
    })

    return {"messages": [AIMessage(content=response.content)]}