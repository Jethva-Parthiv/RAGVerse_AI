from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from app.core.settings import CHAT_MODEL_NAME

load_dotenv()

def get_gemini_chat_model():
    model = ChatGoogleGenerativeAI(model=CHAT_MODEL_NAME)
    return model