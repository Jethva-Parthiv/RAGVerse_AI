def is_exit_command(text: str) -> bool:
    return text.lower() in ["exit", "bye"]