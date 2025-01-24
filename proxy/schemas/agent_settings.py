from typing import Optional

from pydantic import BaseModel, Field, HttpUrl


class AgentSetting(BaseModel):
    organization_id: int
    version_id: int
    core_llm: str = Field(max_length=20)
    condense_llm: str = Field(max_length=20)
    vision_llm: Optional[str] = Field(default=None, max_length=20)
    embeddings: str = Field(max_length=20)
    delay: bool = False
    hide_tool_messages: bool = False

    class Config:
        orm_mode = True


class Tool(BaseModel):
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
        orm_mode = True
