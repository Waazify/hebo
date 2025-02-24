from enum import Enum
from typing import Optional, List

from pydantic import BaseModel, Field, HttpUrl


class ModelType(str, Enum):
    CHAT = "chat"
    EMBEDDING = "embedding"


class ProviderType(str, Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    AZURE = "azure"
    BEDROCK = "bedrock"
    VERTEX = "vertex-ai"
    VOYAGE = "voyage"


class LLMAdapter(BaseModel):
    id: int
    is_default: bool
    organization_id: Optional[str] = None
    model_type: ModelType
    provider: ProviderType
    api_base: Optional[str] = None
    name: str = Field(max_length=150)
    aws_region: Optional[str] = Field(default=None, max_length=50)
    api_key: Optional[str] = Field(default=None, max_length=2000)
    aws_access_key_id: Optional[str] = Field(default=None, max_length=255)
    aws_secret_access_key: Optional[str] = Field(default=None, max_length=255)

    class Config:
        from_attributes = True


class Tool(BaseModel):
    id: int
    agent_setting_id: int
    name: str = Field(max_length=200)
    description: str
    output_template: str
    tool_type: str = Field(max_length=20)
    openapi_url: Optional[HttpUrl] = None
    auth_token: Optional[str] = Field(default=None, max_length=255)
    db_connection_string: Optional[str] = Field(default=None, max_length=255)
    query: Optional[str] = None

    class Config:
        from_attributes = True


class AgentSetting(BaseModel):
    id: int
    organization_id: str
    version_id: int
    core_llm: Optional[LLMAdapter] = None
    condense_llm: Optional[LLMAdapter] = None
    vision_llm: Optional[LLMAdapter] = None
    embeddings: Optional[LLMAdapter] = None
    delay: bool = False
    hide_tool_messages: bool = False
    tools: List[Tool] = []

    class Config:
        from_attributes = True
