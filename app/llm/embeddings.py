from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from app.core.settings import GEMINI_EMBEDDING_MODEL_NAME,HUGGINGFACE_EMBEDDING_MODEL_NAME
load_dotenv()

def get_gemini_embedding_model():
    embedding_model = GoogleGenerativeAIEmbeddings(
        model=GEMINI_EMBEDDING_MODEL_NAME
    )
    return embedding_model


def get_huggingface_embedding_model():
    return HuggingFaceEmbeddings(
        model_name=HUGGINGFACE_EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},   # change to "cuda" if you have GPU
        encode_kwargs={
            "normalize_embeddings": True,  # BGE requires this for cosine similarity
            "batch_size": 256,             # no rate limit — use large batches
        },
    )
