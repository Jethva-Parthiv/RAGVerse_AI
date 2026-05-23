from uuid import uuid4


def generate_thread_id() -> str:
    return f"chat-{uuid4().hex[:8]}"