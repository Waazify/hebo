from django.contrib import admin
from django.contrib.admin import ModelAdmin
from .models import Agent, AgentSlug, Version, VersionSlug


@admin.register(Agent)
class AgentAdmin(ModelAdmin):
    list_display = ["name", "slug", "organization", "created_at"]
    search_fields = ["name", "organization__name"]
    list_filter = ["created_at"]


@admin.register(AgentSlug)
class AgentSlugAdmin(ModelAdmin):
    list_display = ["slug", "agent", "created_at"]
    search_fields = ["slug", "agent__name"]
    list_filter = ["created_at"]


@admin.register(Version)
class VersionAdmin(ModelAdmin):
    list_display = ["name", "agent", "status", "version_slugs", "created_at"]
    search_fields = ["name", "agent__name", "slugs__slug"]
    list_filter = ["status", "created_at"]

    def version_slugs(self, obj):
        return ", ".join([slug.slug for slug in obj.slugs.all()])
    version_slugs.short_description = "Version Slugs"


@admin.register(VersionSlug)
class VersionSlugAdmin(ModelAdmin):
    list_display = ["slug", "version", "created_at"]
    search_fields = ["slug", "version__name"]
    list_filter = ["created_at"]
