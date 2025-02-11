import asyncio
import logging
import re
import uuid
from datetime import datetime, timedelta
from typing import List

import asyncpg
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
from db.vectorstore import VectorStore
from .ai.conversations import execute_conversation
from .ai.vision import get_content_from_human_message
from .exceptions import ColleagueHandoffException
from .retriever import Retriever


logger = logging.getLogger(__name__)


class ThreadManager:
    def __init__(
        self,
        conn: asyncpg.Connection,
    ):
        self.db: DB = DB(conn)
        self.vectorstore: VectorStore = VectorStore(conn)

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
        id = await self.db.add_message(message)
        message.id = id
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

    async def remove_message(
        self, message_id: int, thread_id: int, organization_id: str
    ) -> int:
        thread = await self.db.get_thread(thread_id, organization_id)

        if not thread:
            raise HTTPException(status_code=404, detail="Thread not found")

        await self.db.remove_message(message_id, thread_id)
        return message_id

    async def run_thread(
        self, run_request: RunRequest, thread_id: int, organization_id: str
    ):
        logger.info("load conversation detail from DB")
        run_id = None
        thread = await self.db.get_thread(thread_id, organization_id)
        if not thread or thread.id is None:
            logger.info("thread %s not found", thread_id)
            raise HTTPException(status_code=404, detail="Thread not found")
        try:
            # Init the retriever
            agent_settings = await self.db.get_agent_settings(
                run_request.agent_version, organization_id
            )

            if not agent_settings:
                raise HTTPException(status_code=404, detail="Agent settings not found")
            if not agent_settings.core_llm:
                raise HTTPException(status_code=404, detail="Core LLM not found")
            if not agent_settings.embeddings:
                raise HTTPException(status_code=404, detail="Embeddings not found")

            retriever = Retriever(
                vector_store=self.vectorstore, agent_settings=agent_settings
            )
            # Create the run
            run = Run(
                organization_id=organization_id,
                version_id=agent_settings.version_id,
                thread_id=thread_id,
                status=RunStatus.CREATED,
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )
            run_id = await self.db.create_run(run)
            run_response = RunResponse(
                agent_version=run_request.agent_version,
                status=RunStatus.CREATED,
            )
            yield f"data: {run_response.model_dump_json(exclude_none=True)}\n\n"

            messages = await self.db.get_valid_thread_messages(
                thread_id, organization_id
            )
            # TODO: refactor the following: we can reduce code duplication here.
            # TODO: Hint: raise colleague handoff exception if the last message is not human.
            if not messages or messages[-1].message_type != MessageType.HUMAN:
                run_response = RunResponse(
                    agent_version=run_request.agent_version,
                    status=RunStatus.ERROR,
                    message=BaseMessage(
                        message_type=MessageType.COMMENT,
                        content=[
                            MessageContent(
                                type=MessageContentType.ERROR,
                                error="Last message in thread is not human. Skipping processing.",
                            ),
                        ],
                    ),
                )
                await self.db.update_run_status(
                    run_id, RunStatus.ERROR.value, organization_id
                )
                yield f"data: {run_response.model_dump_json(exclude_none=True)}\n\n"
                message = Message(
                    message_type=MessageType.COMMENT,
                    content=[
                        MessageContent(
                            type=MessageContentType.ERROR,
                            error="Last message in thread is not human. Skipping processing.",
                        ),
                    ],
                    created_at=datetime.now(),
                    thread_id=thread.id,
                    run_status=RunStatus.ERROR,
                )
                await self._add_message(message)
                return

            # We include a small latency to make sure the user has finished typing their multi-part message
            # This is more art than science. We should think about a more robust solution in the future. (And patent it eventually)
            # The solution should handle cases where users are sending multi-part messages. A more sophisticated solution for the traffic light is needed.
            await asyncio.sleep(8 if settings.TARGET_ENV == "production" else 1)
            run_status = await self._get_run_status(run_id, organization_id)
            if run_status != RunStatus.CREATED:
                run_response = RunResponse(
                    agent_version=run_request.agent_version,
                    status=run_status,
                )
                yield f"data: {run_response.model_dump_json(exclude_none=True)}\n\n"
                logger.info("Run %s is not in CREATED status", run_id)

            # Merge messages
            conversation_messages = self._merge_sanitize_messages(messages)
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
                agent_version=run_request.agent_version,
                organization_id=organization_id,
            )

            # Retrieve relevant context
            context = await retriever.get_relevant_sources(llm_conversation, session)
            behaviour_part_ids = await self.db.get_behaviour_part_ids(
                agent_settings.version_id, organization_id
            )
            behaviour_parts = []
            for id in behaviour_part_ids:
                part = await self.vectorstore.find_by_id(id)
                if part:
                    behaviour_parts.append(part)
            behaviour_parts = "\n\n".join([part.content for part in behaviour_parts])

            logger.info("conversation has %s messages", len(llm_conversation))
            logger.info("process conversation with LLM")
            logger.debug(
                "Responding to message: %s",
                conversation_messages[-1].content,
            )

            # TODO: retrieve past conversation summaries

            reply_messages = []
            for reply in execute_conversation(
                agent_settings_or_llm=agent_settings,
                conversation=llm_conversation,
                behaviour=behaviour_parts,
                context=context,
                session=session,
            ):
                # Replies from execute_conversation are of type AiMessage or ToolMessage
                # AiMessages may contain tool calls, ToolMessages are 1 single text message
                message_type = (
                    MessageType.AI
                    if isinstance(reply, AIMessage)
                    else MessageType.TOOL_ANSWER
                )
                message_content = []
                for content in reply.content:
                    if isinstance(content, dict):
                        message_content_type = (
                            MessageContentType.TEXT
                            if content.get("type", "") == "text"
                            else MessageContentType.TOOL_USE
                        )
                        if message_content_type == MessageContentType.TOOL_USE:
                            message_content.append(
                                MessageContent(
                                    type=message_content_type,
                                    name=content.get("name", ""),
                                    input=content.get("input", {}),
                                    id=content.get("id", ""),
                                )
                            )
                        else:
                            message_content.append(
                                MessageContent(
                                    type=message_content_type,
                                    text=content.get("text", ""),
                                )
                            )
                    else:
                        message_content.append(
                            MessageContent(
                                type=MessageContentType.TEXT,
                                text=content,
                            )
                        )

                reply = Message(
                    message_type=message_type,
                    content=message_content,
                    thread_id=thread.id,
                    created_at=datetime.now(),
                )

                reply_messages.append(reply)
                run_status = await self._get_run_status(run_id, organization_id)
                should_send = self._should_send_message(
                    reply_messages,
                    llm_conversation,
                    run_status,
                    agent_settings.hide_tool_messages,
                )

                message_parts = self._split_message(reply.content)
                for i, part in enumerate(message_parts):
                    if part.text:
                        if not re.search(r"call \w+ tool", part.text):
                            # Calculate delay based on word count and reading speed
                            word_count = len(part.text.split())
                            # Convert WPM to seconds
                            delay = (word_count / 100) * 60
                            # Add a delay of -15 seconds for the first part to account for the system latency
                            delay -= 15 if i == 0 else 0
                            await asyncio.sleep(
                                max(0, delay) if agent_settings.delay else 0
                            )

                            if (
                                await self._get_run_status(run_id, organization_id)
                                == RunStatus.CREATED
                            ):
                                await self.db.update_run_status(
                                    run_id, RunStatus.RUNNING.value, organization_id
                                )
                                run_response = RunResponse(
                                    agent_version=run_request.agent_version,
                                    status=RunStatus.RUNNING,
                                )
                                yield f"data: {run_response.model_dump_json(exclude_none=True)}\n\n"

                            base_message = BaseMessage(
                                message_type=message_type,
                                content=[part],
                            )
                            logger.info("send message to user")
                            run_status = await self._get_run_status(
                                run_id, organization_id
                            )
                            run_response = RunResponse(
                                agent_version=run_request.agent_version,
                                status=run_status,
                                message=base_message,
                                should_send=should_send,
                            )
                            yield f"data: {run_response.model_dump_json(exclude_none=True)}\n\n"
                            message = Message(
                                message_type=reply.message_type,
                                content=[part],
                                created_at=datetime.now(),
                                thread_id=thread.id,
                                run_status=run_status,
                            )
                            await self._add_message(message)

                        else:
                            logger.warning("Irregular handover from the agent.")
                            raise ColleagueHandoffException(
                                "The agent is handing over the conversation, please read the conversation history carefully."
                            )
                    if part.type == MessageContentType.TOOL_USE:
                        run_status = await self._get_run_status(run_id, organization_id)
                        run_response = RunResponse(
                            agent_version=run_request.agent_version,
                            status=run_status,
                            message=BaseMessage(
                                message_type=MessageType.AI,
                                content=[part],
                            ),
                            should_send=False,
                        )
                        yield f"data: {run_response.model_dump_json(exclude_none=True)}\n\n"
                        message = Message(
                            message_type=MessageType.AI,
                            content=[part],
                            created_at=datetime.now(),
                            thread_id=thread.id,
                            run_status=run_status,
                        )
                        await self._add_message(message)

            run_status = await self._get_run_status(run_id, organization_id)
            if run_status in [RunStatus.RUNNING, RunStatus.CREATED]:
                await self.db.update_run_status(
                    run_id, RunStatus.COMPLETED.value, organization_id
                )
                run_response = RunResponse(
                    agent_version=run_request.agent_version,
                    status=RunStatus.COMPLETED,
                )
                yield f"data: {run_response.model_dump_json(exclude_none=True)}\n\n"
                logger.info("Run %s is completed", run_id)

        except Exception as e:
            logger.error("Error running thread: %s", e, exc_info=True)

            logger.warning("Handing off thread %s because of Exception.")

            run_response = RunResponse(
                agent_version=run_request.agent_version,
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
                        ),
                        MessageContent(
                            type=MessageContentType.ERROR,
                            error=str(e),
                        ),
                    ],
                ),
            )
            if run_id:
                await self.db.update_run_status(
                    run_id, RunStatus.ERROR.value, organization_id
                )
            yield f"data: {run_response.model_dump_json(exclude_none=True)}\n\n"
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
                    ),
                    MessageContent(
                        type=MessageContentType.ERROR,
                        error=str(e),
                    ),
                ],
                created_at=datetime.now(),
                thread_id=thread.id,
                run_status=RunStatus.ERROR,
            )
            await self._add_message(message)
            return

    async def _get_run_status(self, run_id: int, organization_id: str) -> RunStatus:
        run = await self.db.get_run(run_id, organization_id)
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")
        return run.status

    @staticmethod
    def _should_send_message(
        replies: List[Message],
        conversation_messages: List[
            AIMessage | LangchainBaseMessage | HumanMessage | ToolMessage
        ],
        run_status: RunStatus,
        hide_tool_messages: bool = False,
    ) -> bool:

        if run_status in [RunStatus.ERROR, RunStatus.EXPIRED, RunStatus.COMPLETED]:
            return False

        if replies[-1].message_type != MessageType.AI:
            return False

        # The first message is always sent
        if len([m for m in conversation_messages if isinstance(m, AIMessage)]) == 0:
            return True

        if not hide_tool_messages:
            return True

        # We send message between two tool calls, unless the tool call is an error
        if len(replies) > 1 and replies[-2].message_type == MessageType.TOOL_ANSWER:
            if "error" not in [
                c.text.lower() for c in replies[-2].content if c.text is not None
            ] and any(
                tool_call.name
                in [c.text.lower() for c in replies[-2].content if c.text is not None]
                for tool_call in replies[-1].content
            ):
                return True

        return False

    @staticmethod
    def _split_message(content: List[MessageContent]) -> List[MessageContent]:
        response_text = ""

        response_text = str(
            "\n\n".join([item.text for item in content if item.text is not None])
        )

        # Split the response text into parts and format each part as a MessageContent
        return [
            MessageContent(type=MessageContentType.TEXT, text=part)
            for part in response_text.split("\n\n")
        ] + [c for c in content if c.type == MessageContentType.TOOL_USE]

    def _sanitize_messages(self, messages: List[Message]) -> List[Message]:
        """This assumes messages have been sorted and merged"""
        if messages[-1].message_type != MessageType.HUMAN:
            return self._sanitize_messages(messages[:-1])

        if messages[-2].message_type == MessageType.TOOL_ANSWER:
            tool_name = f" {messages[-2].content[0].name}"
            messages[-2].content.append(
                MessageContent(
                    type=MessageContentType.TEXT,
                    text=f"Above is the tool answer{tool_name}. In the meantime, the user has sent the following messages:",
                )
            )
            for message_content in messages[-1].content:
                messages[-2].content.append(message_content)
            messages[-2].content.append(
                MessageContent(
                    type=MessageContentType.TEXT,
                    text="Please continue the conversation considering the above tool answer and the user's messages.",
                )
            )
            return messages[:-1]

        if messages[-2].message_type == MessageType.AI and any(
            c.type == MessageContentType.TOOL_USE for c in messages[-2].content
        ):
            for content in messages[-2].content:
                if content.type == MessageContentType.TOOL_USE:
                    tool_call_id = content.id

                    # Create a tool answer message between the AI tool use and human message
                    tool_answer_message = Message(
                        message_type=MessageType.TOOL_ANSWER,
                        content=[
                            MessageContent(
                                type=MessageContentType.TEXT,
                                text="The tool was called but no response was received. Try again.",
                            )
                        ],
                        created_at=messages[-1].created_at - timedelta(seconds=1),
                        thread_id=messages[-1].thread_id,
                        tool_call_id=tool_call_id,
                    )

                    # Insert the tool answer message between AI and human messages
                    messages.insert(-1, tool_answer_message)
            return self._sanitize_messages(messages)

        return messages

    @staticmethod
    def _merge_messages(messages: List[Message]) -> List[Message]:
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

    def _merge_sanitize_messages(self, messages: List[Message]) -> List[Message]:
        messages = self._merge_messages(messages)
        return self._sanitize_messages(messages)

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
            f"Message type '{content_type}' is not supported. The message will be ignored."
        )

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
            MessageType.TOOL_ANSWER: self._format_tool_message,
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
        content = await get_content_from_human_message(message)
        message.content = content
        message.message_type = MessageType.TOOL_ANSWER
        return message

    @staticmethod
    async def _format_comment_message(message: Message) -> Message:
        content = await get_content_from_human_message(message)
        message.content = content
        message.message_type = MessageType.COMMENT
        return message

    @staticmethod
    async def _format_ai_message(message: Message) -> Message:
        message.message_type = MessageType.AI
        return message
