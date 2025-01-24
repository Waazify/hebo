from typing import Optional


def get_system_prompt(context: Optional[str] = None) -> str:
    """
    Constructs the system prompt with optional context injection.

    Args:
        context: Optional context string

    Returns:
        Formatted system prompt string
    """

    return f"""You are a helpful assistant. Provide an answer to the user's question based on the following context:
{context}
"""
