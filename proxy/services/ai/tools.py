import logging

from langchain_core.tools import tool

from config import settings

logger = logging.getLogger(__name__)

ARTIFICIAL_DELAY_DURATION = settings.ARTIFICIAL_DELAY_DURATION


@tool
def colleague_handoff(english_query: str) -> str:
    """
    Use this tool to hand off the conversation to your colleague.

    Args:
        query: The query in english to hand off to your colleague.
    """
    return f"Tool (colleague handoff): {english_query}"


# TODO: Add tool factory
