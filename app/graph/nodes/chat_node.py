from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from app.graph.state import State
from app.llm.models import get_gemini_chat_model
from app.llm.prompts import SYSTEM_RAG_PROMPT
from app.retrieval.advanced_retriever import get_advanced_retriever
from app.core.logging import logger

_retriever = None
_llm = None
_chain = None


def _get_chain():
    global _retriever, _llm, _chain
    if _chain is None:
        _retriever = get_advanced_retriever()
        _llm = get_gemini_chat_model()

        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_RAG_PROMPT),
            MessagesPlaceholder(variable_name="history"),
            ("human", "Context:\n{context}\n\nQuestion: {query}"),
        ])
        _chain = prompt | _llm
    return _retriever, _chain


def chat_node(state: State) -> dict:
    """
    StateGraph node performing retrieval, context assembly, and Gemini LLM invocation.
    """
    query = state["query"]
    history = state["messages"][:-1]

    retriever, chain = _get_chain()

    try:
        docs = retriever.retrieve(query)
        if docs:
            context = "\n\n---\n\n".join(
                f"[Source: {doc.metadata.get('source_file', 'unknown')}]\n{doc.page_content}"
                for doc in docs
            )
        else:
            context = "No relevant context found in the technical documentation knowledge base."
    except Exception as e:
        logger.error(f"[chat_node] Retrieval failed: {e}")
        context = "Retrieval error occurred."

    response = chain.invoke({
        "history": history,
        "context": context,
        "query": query,
    })

    return {"messages": [AIMessage(content=response.content)]}