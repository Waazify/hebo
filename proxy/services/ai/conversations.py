import logging
from typing import Generator, List, Optional

from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import Runnable

from config import settings
from schemas.ai import Session
from schemas.agent_settings import AgentSetting, Tool
from services.exceptions import ColleagueHandoffException

from .chat_models.bedrock import get_bedrock_client
from .langfuse_utils import get_langfuse_config
from .llms import init_llm
from .prompts.condense import get_condense_prompt
from .prompts.summary import get_summary_prompt
from .prompts.system import get_system_prompt
from .prompts.vision import get_vision_prompt
from .tools import colleague_handoff

MAX_RECURSION_DEPTH = settings.MAX_RECURSION_DEPTH

logger = logging.getLogger(__name__)


# TODO: make this async
def execute_conversation(
    agent_settings_or_llm: AgentSetting | Runnable[LanguageModelInput, BaseMessage],
    conversation: List[BaseMessage],
    session: Session,
    behaviour: str,
    context: str,
    history_summaries: Optional[str] = None,
    tools: Optional[List[Tool]] = None,
    recursion_depth: int = 0,
) -> Generator[AIMessage | BaseMessage | ToolMessage, None, None]:
    """Execute a conversation with the LLM and yield messages to be returned."""

    agent_settings = (
        agent_settings_or_llm
        if isinstance(agent_settings_or_llm, AgentSetting)
        else None
    )
    llm = agent_settings_or_llm if isinstance(agent_settings_or_llm, Runnable) else None
    tools = (
        tools
        if isinstance(tools, List)
        else agent_settings.tools if agent_settings else None
    )

    logger.debug(f"Executing conversation. Recursion depth: {recursion_depth}")

    if recursion_depth >= MAX_RECURSION_DEPTH:
        logger.warning(f"Max recursion depth reached: {recursion_depth}")
        raise ColleagueHandoffException(
            "Agent ran out of time. Please, take over the conversation."
        )

    langfuse_config = get_langfuse_config("conversation", session)

    def get_llm(
        client,
        model_name: str,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        """Initialize the LLM, bind tools to it, and return the instance."""
        logger.debug("Getting LLM instance")

        llm = init_llm(client, model_name)
        # TODO: Add support for user defined tools
        return llm.bind_tools(
            [
                colleague_handoff,
            ]
        )

    if recursion_depth == 0:

        conversation = [
            SystemMessage(content=get_system_prompt(context, behaviour, history_summaries))
        ] + conversation

    if not llm and agent_settings:
        # TODO: Add support for other LLM providers
        conversation_client = get_bedrock_client(
            (
                agent_settings.core_llm.aws_access_key_id
                if agent_settings
                and agent_settings.core_llm
                and agent_settings.core_llm.aws_access_key_id
                else ""
            ),
            (
                agent_settings.core_llm.aws_secret_access_key
                if agent_settings
                and agent_settings.core_llm
                and agent_settings.core_llm.aws_secret_access_key
                else ""
            ),
            (
                agent_settings.core_llm.aws_region
                if agent_settings
                and agent_settings.core_llm
                and agent_settings.core_llm.aws_region
                else ""
            ),
        )

        model_name = agent_settings.core_llm.name if agent_settings.core_llm else None
        if not model_name:
            raise ValueError("Model name not found")

        llm = get_llm(conversation_client, model_name)

    if not llm:
        raise ValueError("LLM not found")

    try:
        logger.info("Invoking Conversation LLM...")
        response = llm.invoke(
            conversation,
            config=langfuse_config,
        )
    except Exception as e:
        logger.error(f"Error invoking LLM: {e}")
        raise
    # we retry because LLMs sometimes return empty response content
    if not response.content:
        logger.warning("LLM response content is empty. Retrying...")
        yield from execute_conversation(
            llm,
            conversation,
            session,
            behaviour,
            context,
            history_summaries,
            tools,
            recursion_depth + 1,
        )
        return

    if isinstance(response.content, str):
        response.content = [{"type": "text", "text": response.content}]
    logger.debug(f"Conversation LLM response: {response}")
    conversation.append(response)
    yield response

    if isinstance(response, AIMessage) and response.tool_calls:
        logger.debug(f"Processing {len(response.tool_calls)} tool calls")
        for tool_call in response.tool_calls:
            logger.info(f"Invoking tool: {tool_call}")
            try:
                response_text = eval(tool_call["name"]).invoke(tool_call["args"])
            except ColleagueHandoffException as e:
                raise e
            except Exception as e:
                logger.warning(f"Error invoking tool {tool_call['name']}: {e}")
                response_text = f"Tool ({tool_call['name']}): Error invoking tool: {e}"

            tool_message_content = [
                {
                    "type": "text",
                    "text": response_text,
                }
            ]

            tool_message = ToolMessage(
                content=tool_message_content,  # type: ignore
                tool_call_id=tool_call["id"],
                additional_kwargs={"tool_call_name": tool_call["name"]},
            )
            conversation.append(tool_message)
            yield tool_message

        # Recursive call
        logger.debug(f"Making recursive call. Current depth: {recursion_depth}")
        yield from execute_conversation(
            llm,
            conversation,
            session,
            behaviour,
            context,
            history_summaries,
            tools,
            recursion_depth + 1,
        )


# TODO: make this async
def execute_vision(
    client,
    conversation: List[BaseMessage],
    session: Session,
    # TODO: agent_settings is used just for the model name. Client depends on AgentSettings.
    # TODO: This should be refactored and implemented in a more elegant way.
    agent_settings: AgentSetting,
) -> str:
    langfuse_config = get_langfuse_config("vision", session)

    def get_llm(client):
        """Initialize the LLM, bind tools to it, and return the instance."""
        logger.debug("Getting LLM instance")
        return init_llm(
            client,
            agent_settings.vision_llm.name if agent_settings.vision_llm else None,
        )

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
    conversation: List[BaseMessage],
    session: Session,
    agent_settings: AgentSetting,
    operation: str = "condense",
) -> List[str]:
    """Format conversation and return tuple of (previous_chat_history, last_message)"""
    if not conversation:
        return [""]

    def format_message(
        message: BaseMessage,
    ) -> str:
        """Helper function to format a single message"""
        first_person = "A" if operation == "condense" else "User"
        second_person = "B" if operation == "condense" else "You (Assistant)"
        prefix = (
            f"{first_person}: "
            if isinstance(message, HumanMessage)
            else f"{second_person}: "
        )

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
    return [
        msg
        for msg in (format_message(message) for message in conversation)
        if msg  # Filter out empty messages
    ]


# TODO: make this async
def execute_condense(
    client,
    conversation: List[BaseMessage],
    session: Session,
    # TODO: agent_settings is used just for the model name. Client depends on AgentSettings.
    # TODO: This should be refactored and implemented in a more elegant way.
    agent_settings: AgentSetting,
) -> str:
    """Execute a conversation with the LLM and yield messages to be returned."""

    langfuse_config = get_langfuse_config("condense", session)

    def get_llm(client):
        """Initialize the LLM, bind tools to it, and return the instance."""
        logger.debug("Getting LLM instance")
        return init_llm(
            client,
            agent_settings.condense_llm.name if agent_settings.condense_llm else None,
        )

    llm = get_llm(client)
    messages = _format_conversation(client, conversation, session, agent_settings)
    chat_history = "\n".join(messages[:-1])
    follow_up_question = messages[-1].replace("B: ", "")

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


# TODO: make this async
def execute_summary(
    agent_settings: AgentSetting,
    conversation: List[BaseMessage],
    session: Session,
):
    langfuse_config = get_langfuse_config("summary", session)

    def get_llm(client):
        """Initialize the LLM, bind tools to it, and return the instance."""
        logger.debug("Getting LLM instance")
        return init_llm(
            client,
            agent_settings.condense_llm.name if agent_settings.condense_llm else None,
        )

    # TODO: Add support for other LLM providers
    client = get_bedrock_client(
        (
            agent_settings.condense_llm.aws_access_key_id
            if agent_settings
            and agent_settings.condense_llm
            and agent_settings.condense_llm.aws_access_key_id
            else ""
        ),
        (
            agent_settings.condense_llm.aws_secret_access_key
            if agent_settings
            and agent_settings.condense_llm
            and agent_settings.condense_llm.aws_secret_access_key
            else ""
        ),
        (
            agent_settings.condense_llm.aws_region
            if agent_settings
            and agent_settings.condense_llm
            and agent_settings.condense_llm.aws_region
            else ""
        ),
    )

    llm = get_llm(client)

    messages = _format_conversation(
        client, conversation, session, agent_settings, operation="summary"
    )

    try:
        logger.info("Invoking Summary LLM...")
        response = llm.invoke(
            [
                SystemMessage(content=get_summary_prompt("\n".join(messages))),
                HumanMessage(
                    content=(
                        "Generate a detailed summary of the conversation. "
                        "Respond with the summary only. "
                        "No comments or other text."
                    )
                ),
            ],
            config=langfuse_config,
        )
    except Exception as e:
        logger.error(f"Error invoking LLM: {e}")
        raise

    logger.debug(f"Summary LLM response: {response}")
    content = response.content
    return content if isinstance(content, str) else ""
