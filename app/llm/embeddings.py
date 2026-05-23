from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from app.core.settings import EMBEDDING_MODEL_NAME
load_dotenv()

def get_gemini_embedding_model():
    embedding_model = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL_NAME
    )
    return embedding_model