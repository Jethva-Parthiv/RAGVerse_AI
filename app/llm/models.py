from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from app.core.config import settings

def get_gemini_chat_model():
    model = ChatGoogleGenerativeAI(
        model=settings.models.chat_model_name,
        n=1
    )
    return model