from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)

from app.graph.state import State
from app.llm.prompts import BASE_RAG_RULES_HOTPOTQA_DATASET
from app.retrieval.retriever import get_retriever
from app.llm.models import get_gemini_chat_model


# Initialize once
chat_model = get_gemini_chat_model()
retriever = get_retriever()


# Create prompt once (better performance)
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        BASE_RAG_RULES_HOTPOTQA_DATASET,
    ),

    MessagesPlaceholder(variable_name="history"),

    (
        "human",
        """
        Context:
        {context}

        Question:
        {query}
        """
    ),
])


def chat_node(state: State) -> dict:
    """
    Retrieve context and generate AI response.
    """

    query = state["query"]
    history = state.get("messages", [])

    # Retrieve documents
    documents = retriever.invoke(query)

    # Build clean context
    context = "\n\n".join(
        doc.page_content.strip()
        for doc in documents
        if doc.page_content
    )

    # Create chain
    chain = prompt | chat_model

    # Generate response
    response = chain.invoke({
        "history": history,
        "context": context,
        "query": query,
    })

    return {
        "messages": [response]
    }