from django.contrib import admin
from django.contrib.admin import ModelAdmin
from .models import APIKey


@admin.register(APIKey)
class APIKeyAdmin(ModelAdmin):
    list_display = ["organization", "name", "is_active"]
    search_fields = ["organization__name", "name"]
