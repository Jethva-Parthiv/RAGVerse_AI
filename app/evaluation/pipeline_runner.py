
# ── Step 2: Run RAG pipeline ──────────────────────────────────────────────────
from app.llm.prompts import BASE_RAG_RULES_HOTPOTQA_DATASET
from langchain_core.prompts import ChatPromptTemplate
import time
from app.core.settings import *
from app.llm.models import get_gemini_chat_model
from app.retrieval.retriever import get_retriever
from app.evaluation.config import *

def run_rag_pipeline(pairs: list[dict]) -> list[dict]:

    retriever = get_retriever()
    llm       = get_gemini_chat_model()

    prompt = ChatPromptTemplate.from_messages([
        ("system", BASE_RAG_RULES_HOTPOTQA_DATASET),
        ("human", "Context:\n{context}\n\nQuestion: {query}"),
    ])
    chain  = prompt | llm
    total  = len(pairs)
    results = []

    print(f"\n[eval] Running RAG on {total} questions ...\n")

    for i, pair in enumerate(pairs, 1):
        question     = pair["question"]
        ground_truth = pair["ground_truth"]

        # Retrieve
        docs     = retriever.invoke(question)
        contexts = [doc.page_content for doc in docs]
        context  = "\n\n".join(contexts)

        # Generate — Gemini may return list or string in response.content
        try:
            response = chain.invoke({"context": context, "query": question})
            content  = response.content
            if isinstance(content, list):
                answer = " ".join(
                    block.get("text", "") if isinstance(block, dict) else str(block)
                    for block in content
                ).strip()
            else:
                answer = str(content).strip()
        except Exception as e:
            print(f"\n  [eval] Q{i} failed: {e}")
            answer = ""

        if not answer:
            answer = "I don't know."   # RAGAS requires non-empty strings

        results.append({
            "question":     question,
            "ground_truth": ground_truth,
            "answer":       answer,
            "contexts":     contexts,
        })

        icon = "⚠" if answer == "I don't know." else "✓"
        print(f"  [{i:>3}/{total}] {icon} {question[:65]}")
        print(f"           → {answer[:85]}{'...' if len(answer) > 85 else ''}\n")

        time.sleep(SLEEP_BETWEEN)

    return results

