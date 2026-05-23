from langgraph.graph import add_messages
from langchain_core.messages import BaseMessage
from typing import TypedDict,Annotated


class State(TypedDict):
    messages : Annotated[list[BaseMessage],add_messages]
    query : str


# Scalable State For Future 

# class State(TypedDict):
#     messages: Annotated[
#         list[BaseMessage],
#         add_messages
#     ]

#     query: str

#     context: str

#     retrieved_docs: list[str]

#     user_id: str

#     session_id: str

#     intent: str

#     tool_result: str

#     final_answer: str