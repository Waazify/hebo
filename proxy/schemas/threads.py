from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class Thread(BaseModel):
    id: Optional[int] = None
    organization_id: str
    is_open: bool = True
    created_at: datetime
    updated_at: datetime
    contact_name: str = Field(max_length=100)
    contact_identifier: Optional[str] = Field(default=None, max_length=100)

    class Config:
        orm_mode = True


class MessageContentType(Enum):
    TEXT = "text"
    IMAGE = "image"
    TOOL_USE = "tool_use"
    IMAGE_URL = "image_url"
    ERROR = "error"


class MessageContent(BaseModel):
    type: MessageContentType
    text: Optional[str] = None
    image_url: Optional[str] = None
    image_encoding: Optional[str] = None
    # for tool use
    name: Optional[str] = None
    input: Optional[dict] = None
    id: Optional[str] = None
    error: Optional[str] = None


class MessageType(Enum):
    AI = "ai"
    HUMAN = "human"
    HUMAN_AGENT = "human_agent"
    TOOL_ANSWER = "tool"
    COMMENT = "comment"


class RunStatus(Enum):
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"
    EXPIRED = "expired"


class BaseMessage(BaseModel):
    id: Optional[int] = None
    message_type: MessageType
    content: List[MessageContent]

    def to_langchain_format(self) -> dict:
        data = self.model_dump(exclude_none=True)
        if self.content:
            data["content"] = [
                {**c.model_dump(exclude_none=True), "type": c.type.value}
                for c in self.content
            ]
        return data

    @field_validator("content", mode="before")
    @classmethod
    def validate_content(cls, value):
        if not isinstance(value, list):
            raise ValueError("Content must be a list of objects")
        # Convert dict items to MessageContent if needed
        return [
            item if isinstance(item, MessageContent) else MessageContent(**item)
            for item in value
        ]

    class Config:
        validate_assignment = True  # Enable validation on assignment
        ignored_types = (type(None),)  # Handle None values


class Message(BaseMessage):
    thread_id: int
    created_at: datetime
    run_status: Optional[RunStatus] = None
    tool_call_id: Optional[str] = None

    class Config:
        orm_mode = True
        validate_assignment = True  # Enable validation on assignment
        ignored_types = (type(None),)  # Handle None values


class Summary(BaseModel):
    thread_id: int
    content: str
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True


class CreateThreadRequest(BaseModel):
    contact_name: str = Field(..., max_length=100)
    contact_identifier: Optional[str] = Field(None, max_length=100)


class CreateThreadResponse(BaseModel):
    thread_id: Optional[int] = None
    contact_name: str
    contact_identifier: Optional[str] = None
    is_open: bool


class CloseThreadResponse(BaseModel):
    thread_id: int
    is_open: bool


class AddMessageRequest(BaseMessage): ...


class AddMessageResponse(BaseMessage): ...


class RemoveMessageResponse(BaseModel):
    message_id: int


class Run(BaseModel):
    organization_id: str
    version_id: int
    thread_id: int
    status: RunStatus
    created_at: datetime
    updated_at: datetime


class RunRequest(BaseModel):
    agent_version: str


class RunResponse(BaseModel):
    agent_version: str
    status: RunStatus
    message: Optional[BaseMessage] = None
    should_send: bool = False
