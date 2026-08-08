import pytest
from app.graph.state import State
from langchain_core.messages import HumanMessage

def test_state_structure():
    sample_state: State = {
        "messages": [HumanMessage(content="Explain LangGraph checkpointers")],
        "query": "Explain LangGraph checkpointers"
    }
    assert sample_state["query"] == "Explain LangGraph checkpointers"
    assert len(sample_state["messages"]) == 1
