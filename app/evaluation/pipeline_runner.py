import time
from langchain_core.prompts import ChatPromptTemplate

from app.llm.prompts import SYSTEM_RAG_PROMPT
from app.llm.models import get_gemini_chat_model
from app.retrieval.retriever import get_retriever
from app.evaluation.config import SLEEP_BETWEEN
from app.core.logging import logger


def run_rag_pipeline(pairs: list[dict]) -> list[dict]:
    """
    Executes the RAG pipeline over a test set of question-ground_truth pairs.
    """
    retriever = get_retriever()
    llm = get_gemini_chat_model()

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_RAG_PROMPT),
        ("human", "Context:\n{context}\n\nQuestion: {query}"),
    ])
    chain = prompt | llm
    total = len(pairs)
    results = []
    cannot_count = 0

    logger.info(f"[eval] Running RAG pipeline evaluation on {total} questions ...")

    for i, pair in enumerate(pairs, 1):
        question = pair["question"]
        ground_truth = pair["ground_truth"]

        try:
            docs = retriever.invoke(question)
            contexts = [doc.page_content for doc in docs]
            context = "\n\n---\n\n".join(contexts) if contexts else "No context found."
        except Exception as e:
            logger.error(f"[eval] Q{i} retrieval error: {e}")
            docs = []
            contexts = []
            context = "Error during retrieval."

        try:
            response = chain.invoke({"context": context, "query": question})
            content = response.content
            if isinstance(content, list):
                answer = " ".join(
                    block.get("text", "") if isinstance(block, dict) else str(block)
                    for block in content
                ).strip()
            else:
                answer = str(content).strip()
        except Exception as e:
            logger.error(f"[eval] Q{i} LLM generation error: {e}")
            answer = ""

        if not answer:
            answer = "I cannot answer this question based on the provided technical documentation."

        cannot = any(p in answer.lower() for p in [
            "cannot answer", "not contain", "no information", "not provided"
        ])
        if cannot:
            cannot_count += 1

        results.append({
            "question": question,
            "ground_truth": ground_truth,
            "answer": answer,
            "contexts": contexts,
        })

        logger.info(f"[{i}/{total}] Question: {question[:50]}... | Answer: {answer[:50]}...")
        time.sleep(SLEEP_BETWEEN)

    logger.info(f"Pipeline Evaluation Stats: Total={total}, Unanswered={cannot_count}")
    return results