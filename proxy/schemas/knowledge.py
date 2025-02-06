from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class Page(BaseModel):
    organization_id: int
    version_id: int
    title: str = Field(max_length=200)
    content: str
    created_at: datetime
    updated_at: datetime
    is_published: bool = False
    parent_id: Optional[int] = None

    class Config:
        orm_mode = True


class ContentType(Enum):
    """Available content types."""

    BEHAVIOUR = "behaviour"
    SCENARIO = "scenario"
    EXAMPLE = "example"


class Part(BaseModel):
    page_id: int
    start_line: int
    end_line: int
    content_hash: str = Field(max_length=64)
    content_type: ContentType
    identifier: str = Field(max_length=100)
    is_handover: bool = False
    created_at: datetime
    updated_at: datetime
    is_valid: bool = True

    @field_validator("end_line", mode="before")
    @classmethod
    def validate_end_line(cls, value, values):
        if "start_line" in values and value <= values["start_line"]:
            raise ValueError("End line must be greater than start line")
        return value

    @field_validator("is_handover", mode="before")
    @classmethod
    def validate_is_handover(cls, value, values):
        if value and values.get("content_type") == ContentType.BEHAVIOUR:
            raise ValueError(
                "Handover tag can only be applied to scenarios and examples"
            )
        return value

    class Config:
        orm_mode = True


class Vector(BaseModel):
    part_id: int
    embedding_model: str = Field(max_length=20)
    vector: List[float]
    created_at: datetime
    updated_at: datetime
    metadata: Dict[str, Any]

    @field_validator("vector", mode="before")
    @classmethod
    def validate_vector(cls, value, values):
        # Assuming DIMENSION_MAP is available in the context
        dimension_map = {
            "ada002": 1536,
            "minilm": 384,
            "mpnet": 768,
            "bger": 1024,
        }
        if "embedding_model" in values:
            expected_dims = dimension_map.get(values["embedding_model"])
            if expected_dims and len(value) != expected_dims:
                raise ValueError(
                    f"Vector must have {expected_dims} dimensions for {values['embedding_model']}"
                )
        return value

    class Config:
        orm_mode = True
