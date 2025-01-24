import logging
from typing import Any, Generator, List, Optional, Union

from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import Runnable

from ai.tools import colleague_handoff
from config import settings
from llms import init_llm
from schemas.ai import Session
from schemas.agent_settings import AgentSetting, Tool
from services.exceptions import ColleagueHandoffException

from .langfuse_utils import get_langfuse_config
from .prompts.condense import get_condense_prompt
from .prompts.system import get_system_prompt
from .prompts.vision import get_vision_prompt

MAX_RECURSION_DEPTH = settings.MAX_RECURSION_DEPTH

logger = logging.getLogger(__name__)


def execute_conversation(
    client: Any | Runnable[LanguageModelInput, BaseMessage],
    conversation: List[
        AIMessage | BaseMessage | HumanMessage | SystemMessage | ToolMessage
    ],
    session: Session,
    agent_settings: AgentSetting,
    tools: List[Tool] | None = None,
    context: Optional[str] = None,
    recursion_depth: int = 0,
) -> Generator[
    AIMessage | BaseMessage | HumanMessage | SystemMessage | ToolMessage, None, None
]:
    """Execute a conversation with the LLM and yield messages to be returned."""

    logger.debug(f"Executing conversation. Recursion depth: {recursion_depth}")

    if recursion_depth >= MAX_RECURSION_DEPTH:
        logger.warning(f"Max recursion depth reached: {recursion_depth}")
        raise ColleagueHandoffException(
            "Gato ran out of time. Please, take over the conversation."
        )

    langfuse_config = get_langfuse_config("conversation", session)

    def get_llm(
        client: Any | Runnable[LanguageModelInput, BaseMessage]
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        """Initialize the LLM, bind tools to it, and return the instance."""
        logger.debug("Getting LLM instance")
        if isinstance(client, Runnable):
            return client

        llm = init_llm(client, agent_settings.core_llm)
        # TODO: Add support for user defined tools
        return llm.bind_tools(
            [
                colleague_handoff,
            ]
        )

    if recursion_depth == 0:
        for i, msg in enumerate(conversation):
            if isinstance(msg, HumanMessage):
                original_content = msg.content
                if isinstance(original_content, list):
                    original_content.insert(
                        0, {"text": "(first message)", "type": "text"}
                    )
                    conversation[i] = HumanMessage(content=original_content)
                else:
                    conversation[i] = HumanMessage(
                        content=f"(first message) {original_content}"
                    )
                break

        conversation = [
            SystemMessage(content=get_system_prompt(context))
        ] + conversation

    llm = get_llm(client)

    try:
        logger.info("Invoking Conversation LLM...")
        response = llm.invoke(
            conversation,
            config=langfuse_config,
        )
    except Exception as e:
        logger.error(f"Error invoking LLM: {e}")
        raise
    # we retry because Bedrock sometimes returns empty response content
    if not response.content:
        logger.warning("LLM response content is empty. Retrying...")
        yield from execute_conversation(
            llm,
            conversation,
            session,
            agent_settings,
            tools,
            context,
            recursion_depth + 1,
        )
        return

    logger.debug(f"Conversation LLM response: {response}")
    conversation.append(response)
    yield response

    if isinstance(response, AIMessage) and response.tool_calls:
        logger.debug(f"Processing {len(response.tool_calls)} tool calls")
        for tool_call in response.tool_calls:
            # Create dummy tool messages to be used in the next step that will eventually keep the
            # chat history consistent even though the run expires.
            tool_message = ToolMessage(
                content=f"Tool ({tool_call['name']}): Execution has been aborted. Please, invoke the tool again if still required.",
                tool_call_id=tool_call["id"],
            )
            conversation.append(tool_message)
            yield tool_message

        for tool_call in response.tool_calls:
            logger.info(f"Invoking tool: {tool_call}")
            try:
                response_text = eval(tool_call["name"]).invoke(tool_call["args"])
            except ColleagueHandoffException as e:
                raise e
            except Exception as e:
                logger.warning(f"Error invoking tool {tool_call['name']}: {e}")
                response_text = f"Tool ({tool_call['name']}): Error invoking tool: {e}"

            # Find and replace the dummy tool message with the same tool_call_id
            for i, msg in enumerate(conversation):
                if isinstance(msg, ToolMessage) and msg.tool_call_id == tool_call["id"]:
                    conversation[i] = ToolMessage(
                        content=response_text, tool_call_id=tool_call["id"]
                    )
                    yield conversation[i]
                    break

        # Recursive call
        logger.debug(f"Making recursive call. Current depth: {recursion_depth}")
        yield from execute_conversation(
            llm,
            conversation,
            session,
            agent_settings,
            tools,
            context,
            recursion_depth + 1,
        )


def execute_vision(
    client,
    conversation: List[AIMessage | HumanMessage],
    session: Session,
    agent_settings: AgentSetting,
) -> str:
    langfuse_config = get_langfuse_config("vision", session)

    def get_llm(client):
        """Initialize the LLM, bind tools to it, and return the instance."""
        logger.debug("Getting LLM instance")
        return init_llm(client, agent_settings.vision_llm)

    llm = get_llm(client)

    try:
        logger.info("Invoking Vision LLM...")
        messages = [SystemMessage(content=get_vision_prompt())] + conversation
        response = llm.invoke(
            messages,
            config=langfuse_config,
        )
    except Exception as e:
        logger.error(f"Error invoking LLM: {e}")
        raise

    logger.info(f"Vision LLM response: {response}")
    content = response.content
    return content if isinstance(content, str) else ""


def _format_conversation(
    client,
    conversation: List[AIMessage | HumanMessage],
    session: Session,
    agent_settings: AgentSetting,
) -> tuple[str, str]:
    """Format conversation and return tuple of (previous_chat_history, last_message)"""
    if not conversation:
        return "", ""

    def format_message(message: Union[HumanMessage, AIMessage]) -> str:
        """Helper function to format a single message"""
        prefix = "A: " if isinstance(message, HumanMessage) else "B: "

        if isinstance(message.content, list):
            content = ""
            for item in message.content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        content += f"{item['text']}\n"
                    elif item.get("type") == "image_url" and isinstance(
                        message, HumanMessage
                    ):
                        content += f"{execute_vision(client, [message], session, agent_settings)}\n"
            return f"{prefix}{content.replace('\n', ' ')}" if content else ""
        else:
            return f"{prefix}{message.content.replace('\n', ' ')}"

    # Process all messages and join them with newlines
    formatted_messages = [
        msg
        for msg in (format_message(message) for message in conversation[:-1])
        if msg  # Filter out empty messages
    ]

    return "\n".join(formatted_messages), format_message(conversation[-1]).replace(
        "B: ", ""
    )


def execute_condense(
    client,
    conversation: List[AIMessage | HumanMessage],
    session: Session,
    agent_settings: AgentSetting,
) -> str:
    """Execute a conversation with the LLM and yield messages to be returned."""

    langfuse_config = get_langfuse_config("condense", session)

    def get_llm(client):
        """Initialize the LLM, bind tools to it, and return the instance."""
        logger.debug("Getting LLM instance")
        return init_llm(client, agent_settings.condense_llm)

    llm = get_llm(client)
    chat_history, follow_up_question = _format_conversation(
        client, conversation, session, agent_settings
    )

    try:
        logger.info("Invoking Condense LLM...")
        response = llm.invoke(
            [
                SystemMessage(
                    content=get_condense_prompt(chat_history, follow_up_question)
                ),
                HumanMessage(
                    content=(
                        "What is the standalone question? "
                        "Respond with the question only. "
                        "No comments or other text. "
                        "If only one question is present, respond with that question."
                    )
                ),
            ],
            config=langfuse_config,
        )
    except Exception as e:
        logger.error(f"Error invoking LLM: {e}")
        raise

    logger.debug(f"Condense LLM response: {response}")
    content = response.content
    return content if isinstance(content, str) else ""
