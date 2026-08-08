import pytest
from app.llm.prompts import SYSTEM_RAG_PROMPT, RAG_CHAT_PROMPT_TEMPLATE

def test_system_prompt_contains_grounding_rules():
    assert "STRICT GROUNDING & ACCURACY RULES" in SYSTEM_RAG_PROMPT
    assert "LangChain" in SYSTEM_RAG_PROMPT

def test_rag_chat_prompt_template():
    formatted = RAG_CHAT_PROMPT_TEMPLATE.format(
        rules=SYSTEM_RAG_PROMPT,
        history="Human: Hello",
        context="Sample documentation context",
        query="What is LangGraph?"
    )
    assert "Sample documentation context" in formatted
    assert "What is LangGraph?" in formatted
