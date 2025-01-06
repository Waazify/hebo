from django.contrib import admin
from django.contrib.admin import ModelAdmin
from .models import Agent, Version


@admin.register(Agent)
class AgentAdmin(ModelAdmin):
    list_display = ['name', 'organization', 'created_at']
    search_fields = ['name', 'organization__name']
    list_filter = ["created_at"]


@admin.register(Version)
class VersionAdmin(ModelAdmin):
    list_display = ["name", "agent", "status", "created_at"]
    search_fields = ["name", "agent__name"]
    list_filter = ["status", "created_at"]
