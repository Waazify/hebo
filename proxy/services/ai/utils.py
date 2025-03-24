from typing import Any
from langchain_core.messages import AIMessage


def clean_run_manager(obj: Any) -> Any:
    """Recursively remove run_manager from any nested structure."""
    if isinstance(obj, dict):
        return {k: clean_run_manager(v) for k, v in obj.items() if k != "run_manager"}
    elif isinstance(obj, list):
        return [clean_run_manager(item) for item in obj]
    return obj


def clean_ai_message(message: AIMessage) -> AIMessage:
    """Create a clean copy of AIMessage without run_manager in tool_calls."""
    # Create a deep copy to avoid modifying the original
    message_dict = message.model_dump()

    # Clean tool_calls if present
    if "tool_calls" in message_dict:
        message_dict["tool_calls"] = clean_run_manager(message_dict["tool_calls"])

    # Create new AIMessage instance with cleaned data
    return AIMessage(**message_dict)
