from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, RootModel, field_validator


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    DEVELOPER = "developer"
    FUNCTION = "function"


class FunctionCall(BaseModel):
    name: str
    arguments: str


class Message(BaseModel):
    role: MessageRole
    content: Optional[str] = None
    name: Optional[str] = None
    function_call: Optional[FunctionCall] = None


class ToolType(str, Enum):
    WEB_SEARCH_PREVIEW = "web_search_preview"
    FILE_SEARCH = "file_search"
    COMPUTER_USE_PREVIEW = "computer_use_preview"


class BuiltInTool(BaseModel):
    type: ToolType
    vector_store_ids: Optional[List[str]] = None
    display_width: Optional[int] = None
    display_height: Optional[int] = None
    environment: Optional[str] = None


class CustomTool(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Dict[str, Any]


class Tool(RootModel):
    root: Union[BuiltInTool, CustomTool]


class LogProbs(BaseModel):
    tokens: List[str]
    token_logprobs: List[float]
    top_logprobs: List[Optional[Dict[str, float]]]
    text_offset: List[int]


class Choice(BaseModel):
    index: int
    message: Message
    finish_reason: Optional[str] = Field(
        default=None, pattern="^(stop|length|function_call)$"
    )
    logprobs: Optional[LogProbs] = None


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    tool_usage: Optional[Dict[str, Any]] = None


class Error(BaseModel):
    message: str
    type: str
    param: Optional[str] = None
    code: Optional[str] = None


class ResponseRequest(BaseModel):
    model: str
    input: Optional[Union[str, List[str]]] = None
    messages: Optional[List[Message]] = None
    instructions: Optional[str] = None
    # Thread ID for us.
    previous_response_id: Optional[str] = None
    # Delete thread after processing in case this is false.
    store: bool = False
    # Not supported at the moment.
    max_output_tokens: Optional[int] = Field(None, ge=1)
    # Not supported at the moment.
    temperature: float = Field(1.0, ge=0, le=2)
    # Not supported at the moment.
    top_p: float = Field(1.0, ge=0, le=1)
    # Not supported at the moment.
    n: int = Field(1, ge=1, le=1)
    # Not supported at the moment.
    stop: Optional[Union[str, List[str]]] = None
    # Not supported at the moment.
    presence_penalty: float = Field(0.0, ge=-2.0, le=2.0)
    # Not supported at the moment.
    frequency_penalty: float = Field(0.0, ge=-2.0, le=2.0)
    # Not supported at the moment.
    logit_bias: Optional[Dict[str, float]] = None
    # Not supported at the moment.
    logprobs: Optional[Union[int, bool]] = None
    # Not supported at the moment.
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    # Not supported at the moment.
    tools: Optional[List[Tool]] = None
    # Not supported at the moment.
    parallel_tool_calls: bool = False
    # Not supported at the moment.
    truncation: str = Field(default="disabled", pattern="^(auto|disabled)$")
    # Not supported at the moment.
    include: Optional[List[str]] = None
    # Not supported at the moment.
    reasoning: Optional[Dict[str, Any]] = None
    # Not supported at the moment.
    user: Optional[str] = None
    # Since the response endpoint is used by the testing we will support only stream to False at the moment.
    stream: bool = False

    @field_validator("input", "messages")
    @classmethod
    def validate_input_or_messages(cls, v, info):
        if info.data.get("input") is not None and info.data.get("messages") is not None:
            raise ValueError("Only one of 'input' or 'messages' can be provided")
        return v


class Response(BaseModel):
    id: str
    object: str = "response"
    created: int
    model: str
    previous_response_id: Optional[str] = None
    choices: List[Choice]
    usage: Usage
    error: Optional[Error] = None

    class Config:
        from_attributes = True
