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


class MessageContent(BaseModel):
    type: MessageContentType
    text: Optional[str] = None
    image_url: Optional[str] = None
    image_encoding: Optional[str] = None
    # for tool use
    name: Optional[str] = None
    input: Optional[dict] = None
    id: Optional[str] = None


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
    message_type: MessageType
    content: List[MessageContent]

    @field_validator("content", mode="before")
    @classmethod
    def validate_content(cls, value):
        if not isinstance(value, list):
            raise ValueError("Content must be a list of objects")
        for item in value:
            if not isinstance(item, MessageContent):
                raise ValueError("Each content item must be an object")
            if item.type is None:
                raise ValueError("Each content item must have a 'type' field")
        return value


class Message(BaseMessage):
    thread_id: int
    created_at: datetime
    run_status: Optional[RunStatus] = None
    tool_call_id: Optional[str] = None

    class Config:
        orm_mode = True


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


class Run(BaseModel):
    version_id: str
    thread_id: int
    status: RunStatus
    created_at: datetime
    updated_at: datetime


class RunRequest(BaseModel):
    version_id: str


class RunResponse(BaseModel):
    version_id: str
    status: RunStatus
    message: Optional[BaseMessage] = None
    should_send: bool = False
