import asyncio
import logging
import uuid
from datetime import datetime, timedelta
from typing import List, Optional

from fastapi import HTTPException
from langchain_core.messages import (
    AIMessage,
    BaseMessage as LangchainBaseMessage,
    HumanMessage,
    ToolMessage,
)

from config import settings
from db.database import DB
from schemas.ai import Session
from schemas.threads import (
    AddMessageRequest,
    BaseMessage,
    CreateThreadRequest,
    Message,
    MessageContent,
    MessageContentType,
    MessageType,
    Run,
    RunRequest,
    RunResponse,
    RunStatus,
    Thread,
)
from utils import generate_id
from .ai.conversations import execute_conversation
from .ai.vision import get_content_from_human_message
from .exceptions import ColleagueHandoffException
from .retriever import Retriever

logger = logging.getLogger(__name__)


class ThreadManager:
    def __init__(
        self,
        db: DB,
        retriever: Optional[Retriever],
    ):
        self.db: DB = db
        self.retriever: Optional[Retriever] = retriever

    async def create_thread(
        self, request: CreateThreadRequest, organization_id: str
    ) -> Thread:
        """Create a new thread"""
        thread = Thread(
            organization_id=organization_id,
            is_open=True,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            contact_name=request.contact_name,
            contact_identifier=request.contact_identifier,
        )
        thread_id = await self.db.create_thread(thread)
        thread.id = thread_id

        logger.info(
            "Created new thread %s for organization %s", thread_id, organization_id
        )
        return thread

    async def close_thread(self, thread_id: int, organization_id: str) -> Thread | None:
        await self.db.close_thread(thread_id, organization_id)
        thread = await self.db.get_thread(thread_id, organization_id)
        logger.info("Thread %s closed", thread_id)
        return thread

    async def _add_message(self, message: Message) -> Message:
        message = await self._format_message(message)
        await self.db.add_message(message)
        logger.info("added message to thread %s", message.thread_id)
        return message

    async def add_message(
        self,
        message_request: AddMessageRequest,
        thread_id: int,
        organization_id: str,
    ) -> Message:

        thread = await self.db.get_thread(thread_id, organization_id)

        if not thread:
            raise HTTPException(status_code=404, detail="Thread not found")

        for content in message_request.content:
            content_type = self._get_content_type(content)
            if content_type not in ["text", "image"]:
                return await self._handle_unsupported_message(content_type, thread_id)
        message = Message(
            message_type=message_request.message_type,
            content=message_request.content,
            thread_id=thread_id,
            created_at=datetime.now(),
        )

        return await self._add_message(message)

    async def run_thread(
        self, run_request: RunRequest, thread_id: int, organization_id: str
    ):
        logger.info("load conversation detail from DB")
        thread = await self.db.get_thread(thread_id, organization_id)
        if not thread or thread.id is None:
            logger.info("thread %s not found", thread_id)
            raise HTTPException(status_code=404, detail="Thread not found")
        try:
            # Create the run
            run = Run(
                version_id=run_request.version_id,
                thread_id=thread_id,
                status=RunStatus.CREATED,
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )
            run_id = await self.db.create_run(run)
            run_response = RunResponse(
                version_id=run_request.version_id,
                status=RunStatus.CREATED,
            )
            yield f"data: {run_response.model_dump_json()}\n\n"

            messages = await self.db.get_thread_messages(thread_id, organization_id)
            if not messages or messages[-1].message_type != MessageType.HUMAN:
                run_response = RunResponse(
                    version_id=run_request.version_id,
                    status=RunStatus.ERROR,
                    message=BaseMessage(
                        message_type=MessageType.COMMENT,
                        content=[
                            MessageContent(
                                type=MessageContentType.TEXT,
                                text="Last message in thread is not human. Skipping processing.",
                            )
                        ],
                    ),
                )
                yield f"data: {run_response.model_dump_json()}\n\n"
                message = Message(
                    message_type=MessageType.COMMENT,
                    content=[
                        MessageContent(
                            type=MessageContentType.TEXT,
                            text="Last message in thread is not human. Skipping processing.",
                        )
                    ],
                    created_at=datetime.now(),
                    thread_id=thread.id,
                )
                await self._add_message(message)
                return

            # We include a small latency to make sure the user has finished typing their multi-part message
            # This is more art than science. We should think about a more robust solution in the future. (And patent it eventually)
            # The solution should handle cases where users are sending multi-part messages. A more sophisticated solution for the traffic light is needed.
            await asyncio.sleep(8 if settings.TARGET_ENV == "production" else 1)
            run = await self.db.get_run(run_id, organization_id)
            if not run:
                raise Exception("Run not found")
            if run.status != RunStatus.CREATED:
                run_response = RunResponse(
                    version_id=run_request.version_id,
                    status=run.status,
                )
                yield f"data: {run_response.model_dump_json()}\n\n"
                logger.info("Run %s is not in CREATED status", run_id)

            # Merge messages
            conversation_messages = self._merge_conversation_messages(messages)
            llm_conversation: List[
                AIMessage | LangchainBaseMessage | HumanMessage | ToolMessage
            ] = []
            for message in conversation_messages:
                if message.message_type.value in ["ai", "human", "tool"]:
                    message_class = {
                        "ai": AIMessage,
                        "human": HumanMessage,
                        "tool": ToolMessage,
                    }[message.message_type.value]
                    llm_conversation.append(message_class(**message.content))

            # Create a session to trace the thread execution
            session = Session(
                contact_identifier=thread.contact_identifier,
                thread_id=str(thread.id),
                trace_id=uuid.uuid4(),
                agent_version=run_request.version_id,
                organization_id=organization_id,
            )

            agent_settings = await self.db.get_agent_settings(
                run_request.version_id, organization_id
            )

            # Retrieve relevant context
            context = await self.retriever.get_relevant_sources(
                llm_conversation, session
            )

            logger.info("conversation has %s messages", len(llm_conversation))
            logger.info("process conversation with LLM")
            logger.debug(
                "Responding to message: %s",
                conversation_messages[-1].content["object"]["content"],
            )

            reply_messages = []
            any_part_sent = False
            for reply in execute_conversation(
                conversation=llm_conversation,
                context=context,
                client=self.bedrock_client,
                session=session,
            ):
                # This ensures that a chunk of replies are saved in the database with very close timestamps
                # It is in place to reduce the risk that a user message is inserted between an AI message with tool use and the subsequent tool reply
                # The solution is far from perfect but should work for now
                # TODO: Implement a more robust solution to handle this
                first_created_at += timedelta(microseconds=1)

                reply = Message(
                    id=generate_id(),  # ID will be changed when reply is sent
                    content={
                        "object": reply.model_dump(
                            exclude=set(["usage_metadata", "response_metadata"]),
                            exclude_none=True,
                        ),
                        "type": "object",
                    },
                    conversation_id=conversation.id,
                    sender_id=settings.GATO_RESPOND_USER_ID,
                    created_at=first_created_at,
                    message_type="ai" if isinstance(reply, AIMessage) else "tool",
                    receiver_id=conversation.contact_id,
                )
                original_content = reply.content
                reply_messages.append(reply)

                if self._should_send_message(reply_messages, llm_conversation):
                    message_parts = self._split_message(reply.content["object"])
                    for i, part in enumerate(message_parts):
                        if (
                            ("call colleague_handoff tool" not in part["text"])
                            and ("call check_availability tool" not in part["text"])
                            and ("call check_instalment_plans tool" not in part["text"])
                            and (
                                "call check_sale_or_trade_in_value tool"
                                not in part["text"]
                            )
                            and ("call find_closest_store tool" not in part["text"])
                        ):
                            # Calculate delay based on word count and reading speed
                            word_count = len(part["text"].split())
                            # Convert WPM to seconds
                            delay = (word_count / 100) * 60
                            # Add a delay of -15 seconds for the first part to account for the system latency
                            delay -= 15 if i == 0 else 0
                            await asyncio.sleep(
                                max(0, delay)
                                if settings.TARGET_ENV == "production"
                                else 1
                            )

                            # Only check for new messages if we're not in the middle of sending a multi-part message
                            if not any_part_sent:
                                current_last_message_id = (
                                    await self.db.get_last_human_message_id(
                                        conversation.id
                                    )
                                )
                                if (
                                    current_last_message_id
                                    and current_last_message_id != last_human_message_id
                                ):
                                    logger.info(
                                        "New human message detected, skipping AI response",
                                    )
                                    break

                            reply.content = part
                            if (
                                await self.db.get_conversation_assignee_id(
                                    conversation.id
                                )
                                != settings.GATO_RESPOND_USER_ID
                            ):
                                logger.info(
                                    "Detected change of assignee. Skipping AI response.",
                                )
                                return
                            logger.info("send message to user")
                            await self.db.update_conversation_status(
                                conversation.id, "writing"
                            )
                            await self.messenger.send_message(reply)
                            any_part_sent = True
                            logger.info("reply %s sent to user", reply.id)

                        else:
                            reply.content = {
                                "type": "text",
                                "text": "Gato is handing over the conversation, please read the conversation history carefully.",
                            }
                            await self.messenger.handle_handoff(reply)
                            logger.warning("Irregular handover from Gato.")
                            break

                elif reply.message_type in ["tool", "error"]:
                    if (
                        reply.message_type == "error"
                        or "colleague handoff" in reply.content["object"]["content"]
                    ):
                        if reply.content["type"] == "object":
                            reply.content = {
                                "type": "text",
                                "text": reply.content["object"]["content"],
                            }
                        await self.messenger.handle_handoff(reply)
                    reply = self._format_tool_message(reply)

                # In case the reply was split into multiple messages, we restore the original content
                # The id used to save the reply in the DB is the last one used to send the message on Respond io
                reply.content = original_content

            # We only store the entire chunk of replies only if any of its parts has been sent to the user
            if any_part_sent:
                for reply in reply_messages:
                    await self.db.store_message(reply)
            await self.db.update_conversation_status(conversation.id, "ready")

            if self.evaluation_manager and any_part_sent:
                # Use the trace_id of the last AI message for evaluation
                last_ai_message = next(
                    (
                        msg
                        for msg in reversed(reply_messages)
                        if msg.message_type == "ai"
                    ),
                    None,
                )
                if last_ai_message and session.trace_id:
                    await self.evaluation_manager.request_feedback(
                        conversation.id, session.trace_id, contact_id
                    )
        except Exception as e:
            logger.error("Error running thread: %s", e, exc_info=True)

            logger.warning("Handing off thread %s because of Exception.")

            run_response = RunResponse(
                version_id=run_request.version_id,
                status=RunStatus.ERROR,
                message=BaseMessage(
                    message_type=MessageType.COMMENT,
                    content=[
                        MessageContent(
                            type=MessageContentType.TEXT,
                            text=(
                                e.message
                                if isinstance(e, ColleagueHandoffException)
                                else "Something went wrong. Please, take over the conversation."
                            ),
                        )
                    ],
                ),
            )
            yield f"data: {run_response.model_dump_json()}\n\n"
            message = Message(
                message_type=MessageType.COMMENT,
                content=[
                    MessageContent(
                        type=MessageContentType.TEXT,
                        text=(
                            e.message
                            if isinstance(e, ColleagueHandoffException)
                            else "Something went wrong. Please, take over the conversation."
                        ),
                    )
                ],
                created_at=datetime.now(),
                thread_id=thread.id,
            )
            await self._add_message(message)
            return

    @staticmethod
    def _should_send_message(
        replies: List[Message], conversation_messages: List[BaseMessage]
    ) -> bool:
        # We do not return messages in case of colleague handoff invokation
        for reply in replies:
            if reply.message_type == "ai":
                content = AIMessage(**reply.content["object"])
                if hasattr(content, "tool_calls") and content.tool_calls:
                    if any(
                        tool_call["name"] == "colleague_handoff"
                        for tool_call in content.tool_calls
                    ):
                        return False

        if replies[-1].message_type == "ai":
            content = AIMessage(**replies[-1].content["object"])
            if hasattr(content, "tool_calls") and content.tool_calls:
                # We don't show the AI message introducing a tool call, unless it's the very first AI message in the conversation
                if (
                    len([m for m in conversation_messages if isinstance(m, AIMessage)])
                    == 0
                ):
                    return True

                # We send Gato's message between two tool calls, unless the tool call is an error
                if (
                    len(replies) > 1
                    and replies[-2].message_type == "tool"
                    and not (
                        "error" in replies[-2].content["object"]["content"].lower()
                        and any(
                            tool_call["name"]
                            in str(replies[-2].content["object"]["content"]).lower()
                            for tool_call in content.tool_calls
                        )
                    )
                ):
                    return True
                return False
            return True
        return False

    @staticmethod
    def _split_message(content: dict) -> List[dict]:
        response_text = ""

        # Check if the content is a string or a more complex structure
        if isinstance(content["content"], str):
            response_text = str(content["content"])
        else:
            # If it's not a string, join all text items in the content
            response_text = str(
                "\n\n".join(
                    [
                        item.get("text", "")
                        for item in content["content"]
                        if isinstance(item, dict) and "text" in item
                    ]
                )
            )

        # Split the response text into parts and format each part as a dictionary
        return [{"type": "text", "text": part} for part in response_text.split("\n\n")]

    @staticmethod
    def _merge_conversation_messages(messages: List[Message]) -> List[Message]:
        formatted_messages = []
        previous_message = None
        previous_message_type = None
        added_first_human_message = False
        # Sort messages by created_at in ascending order
        messages = sorted(messages, key=lambda x: x.created_at)
        for message in messages:
            if (
                not previous_message_type
                or message.message_type == previous_message_type
            ):
                if previous_message:
                    prev_content = previous_message.content
                    curr_content = message.content
                    message.content = prev_content + curr_content
                elif message.message_type == "human_agent":
                    curr_content = message.content
                    if curr_content[0].type == "text":
                        message.content[0].text = (
                            f"Human colleague: {curr_content[0].text}"
                        )
                    else:
                        message.content.insert(
                            0,
                            MessageContent(
                                type=MessageContentType.TEXT,
                                text="Human colleague: ",
                            ),
                        )
                elif message.message_type == "human":
                    if not added_first_human_message:
                        added_first_human_message = True
                        curr_content = message.content
                        if curr_content[0].type == "text":
                            message.content[0].text = (
                                f"(first message) {curr_content[0].text}"
                            )
                        else:
                            message.content.insert(
                                0,
                                MessageContent(
                                    type=MessageContentType.TEXT,
                                    text="(first message)",
                                ),
                            )

                previous_message = message
                previous_message_type = message.message_type
            else:
                # Add the previous human message (if any) to formatted_messages
                if previous_message:
                    formatted_messages.append(previous_message)
                    previous_message = None
                # Add the non-human message to formatted_messages
                formatted_messages.append(message)

        # Add the last human message if it exists
        if previous_message:
            formatted_messages.append(previous_message)

        return formatted_messages

    @staticmethod
    def _get_content_type(message_content: MessageContent) -> str:
        return message_content.type.value

    async def _handle_unsupported_message(
        self,
        content_type: str,
        thread_id: int,
    ) -> Message:
        # Log the full message object for later investigation
        logger.warning(f"Unexpected message type received: {content_type}. ")

        # Create a concise comment for operations
        comment_text = (
            f"Tool (colleague handoff): Message type '{content_type}' is not supported"
        )

        logger.info("Handoff")
        return Message(
            message_type=MessageType.COMMENT,
            content=[MessageContent(type=MessageContentType.TEXT, text=comment_text)],
            created_at=datetime.now(),
            thread_id=thread_id,
        )

    async def _format_message(self, message: Message) -> Message:

        format_message_map = {
            MessageType.AI: self._format_ai_message,
            MessageType.HUMAN: self._format_human_message,
            MessageType.HUMAN_AGENT: self._format_human_agent_message,
            MessageType.TOOL: self._format_tool_message,
            MessageType.COMMENT: self._format_comment_message,
        }

        return await format_message_map[message.message_type](message)

    @staticmethod
    async def _format_human_message(message: Message) -> Message:
        content = await get_content_from_human_message(message)
        message.content = content
        message.message_type = MessageType.HUMAN
        return message

    @staticmethod
    async def _format_human_agent_message(message: Message) -> Message:
        content = await get_content_from_human_message(message)
        message.content = content
        message.message_type = MessageType.HUMAN_AGENT
        return message

    @staticmethod
    async def _format_tool_message(message: Message) -> Message:
        message.message_type = MessageType.TOOL
        return message

    @staticmethod
    async def _format_comment_message(message: Message) -> Message:
        message.message_type = MessageType.COMMENT
        return message

    @staticmethod
    async def _format_ai_message(message: Message) -> Message:
        message.message_type = MessageType.AI
        return message
