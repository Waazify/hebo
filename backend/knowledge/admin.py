from django.contrib import admin
from django.contrib.admin import ModelAdmin
from .models import Page, Part, VectorStore


@admin.register(Page)
class PageAdmin(ModelAdmin):
    list_display = ["title", "version_slugs", "created_at"]
    search_fields = ["title", "version__name", "version__slugs__slug"]
    list_filter = ["created_at"]

    def version_slugs(self, obj):
        return ", ".join([slug.slug for slug in obj.version.slugs.all()])
    version_slugs.short_description = "Version Slugs"


@admin.register(Part)
class PartAdmin(ModelAdmin):
    list_display = ["page", "start_line", "end_line", "created_at"]
    search_fields = ["page__title", "page__version__name"]
    list_filter = ["created_at"]


@admin.register(VectorStore)
class VectorStoreAdmin(ModelAdmin):
    list_display = ["part", "embedding_model", "created_at"]
    search_fields = ["part__page__title"]
    list_filter = ["created_at"]
