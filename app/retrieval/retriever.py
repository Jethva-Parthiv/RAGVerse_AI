from langchain_community.vectorstores import FAISS
from app.llm.embeddings import get_gemini_embedding_model
from app.core.settings import TOP_K_RESULTS,FAISS_PATH
embedding_model = get_gemini_embedding_model()

def get_retriever():
    db = FAISS.load_local(
        FAISS_PATH,
        embedding_model,
        allow_dangerous_deserialization=True
    )

    return db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K_RESULTS}
    )

