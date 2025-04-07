import json
import logging
from datetime import datetime

import asyncpg
from fastapi import HTTPException

from schemas.responses import (
    ResponseRequest,
    Response,
    Usage,
    Choice,
    Message,
    MessageRole,
)
from schemas.threads import (
    AddMessageRequest,
    CreateThreadRequest,
    MessageType,
    MessageContent,
    MessageContentType,
    RunRequest,
)
from .thread_manager import ThreadManager

logger = logging.getLogger(__name__)


class ResponseManager:
    def __init__(self, conn: asyncpg.Connection):
        self.thread_manager = ThreadManager(conn)

    async def create_response(
        self, request: ResponseRequest, organization_id: str
    ) -> Response:
        """Create a response using the thread manager"""
        thread_id = None
        try:
            # Create or fetch thread
            if request.previous_response_id:
                # TODO: Implement thread fetching by previous_response_id
                thread_id = int(request.previous_response_id)
            else:
                # Create new thread
                thread = await self.thread_manager.create_thread(
                    CreateThreadRequest(
                        contact_name="API User", contact_identifier=None
                    ),
                    organization_id,
                )
                thread_id = thread.id

            if not thread_id:
                raise HTTPException(status_code=500, detail="Failed to get thread ID")

            # Add messages to thread
            if request.input:
                # Handle single input or array of inputs
                inputs = (
                    [request.input] if isinstance(request.input, str) else request.input
                )
                for input_text in inputs:
                    await self.thread_manager.add_message(
                        AddMessageRequest(
                            message_type=MessageType.HUMAN,
                            content=[
                                MessageContent(
                                    type=MessageContentType.TEXT, text=input_text
                                )
                            ],
                        ),
                        thread_id,
                        organization_id,
                    )
            elif request.messages:
                for message in request.messages:
                    if message.role in [MessageRole.SYSTEM, MessageRole.DEVELOPER]:
                        raise HTTPException(
                            status_code=400,
                            detail="SYSTEM and DEVELOPER message roles are not supported",
                        )
                    # Map OpenAI message roles to Hebo message types
                    message_type = {
                        MessageRole.USER: MessageType.HUMAN,
                        MessageRole.ASSISTANT: MessageType.AI,
                        MessageRole.FUNCTION: MessageType.TOOL_ANSWER,
                    }.get(message.role, MessageType.HUMAN)

                    await self.thread_manager.add_message(
                        AddMessageRequest(
                            message_type=message_type,
                            content=[
                                MessageContent(
                                    type=MessageContentType.TEXT,
                                    text=message.content or "",
                                )
                            ],
                        ),
                        thread_id,
                        organization_id,
                    )

            # Run the thread with the specified model
            result = []
            async for response in self.thread_manager.run_thread(
                RunRequest(agent_version=request.model), thread_id, organization_id
            ):
                try:
                    # Parse the response string
                    response_data = response.split("data: ", 1)[1]
                    response_json = json.loads(response_data)

                    # Handle error responses
                    if response_json.get("status") == "error":
                        error_msg = (
                            response_json.get("message", {})
                            .get("content", [{}])[0]
                            .get("error", "Unknown error")
                        )
                        return Response(
                            id=str(thread_id),
                            created=int(datetime.now().timestamp()),
                            model=request.model,
                            choices=[
                                Choice(
                                    index=0,
                                    message=Message(
                                        role=MessageRole.ASSISTANT,
                                        content=f"Error: {error_msg}",
                                    ),
                                    finish_reason="stop",
                                )
                            ],
                            usage=Usage(
                                prompt_tokens=0,
                                completion_tokens=0,
                                total_tokens=0,
                            ),
                        )

                    # Collect text content from successful messages
                    if response_json.get("message") and response_json.get(
                        "should_send", False
                    ):
                        message = response_json["message"]
                        for content in message.get("content", []):
                            if content.get("type") == "text" and content.get("text"):
                                result.append(content["text"])

                except (json.JSONDecodeError, KeyError, IndexError) as e:
                    logger.error(f"Error processing response: {str(e)}")
                    continue

            # Create OpenAI-compatible response
            return Response(
                id=str(thread_id),
                created=int(datetime.now().timestamp()),
                model=request.model,
                choices=[
                    Choice(
                        index=0,
                        message=Message(
                            role=MessageRole.ASSISTANT,
                            content="\n\n".join(result) if result else "",
                        ),
                        finish_reason="stop",
                    )
                ],
                usage=Usage(
                    prompt_tokens=0,  # TODO: Implement token counting
                    completion_tokens=0,
                    total_tokens=0,
                ),
            )

        except Exception as e:
            logger.error(f"Error creating response: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

        finally:
            # Delete thread if store is False
            if not request.store and thread_id:
                try:
                    # TODO: Implement thread deletion
                    pass
                except Exception as e:
                    logger.error(f"Error deleting thread: {str(e)}")
