from django.contrib import admin
from django.contrib.admin import ModelAdmin

from .models import AgentSetting, Tool


@admin.register(AgentSetting)
class AgentSettingAdmin(ModelAdmin):
    list_display = [
        "organization",
        "version",
        "core_llm",
        "condense_llm",
        "embeddings",
        "delay",
        "hide_tool_messages",
    ]
    list_filter = [
        "organization",
        "version",
        "core_llm",
        "condense_llm",
        "embeddings",
        "delay",
        "hide_tool_messages",
    ]
    search_fields = ["organization__name", "version__name"]


@admin.register(Tool)
class ToolAdmin(ModelAdmin):
    list_display = ["name", "description", "tool_type"]
    search_fields = ["name", "description"]
