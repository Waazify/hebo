from django.contrib import admin
from django.contrib.admin import ModelAdmin
from .models import Message, Thread, Run, Summary


@admin.register(Message)
class MessageAdmin(ModelAdmin):
    list_display = ["id", "thread", "created_at"]
    search_fields = ["thread__contact_name"]
    list_filter = ["created_at"]


@admin.register(Thread)
class ThreadAdmin(ModelAdmin):
    list_display = ["id", "contact_name", "created_at"]
    search_fields = ["contact_name"]
    list_filter = ["created_at"]


@admin.register(Run)
class RunAdmin(ModelAdmin):
    list_display = ["id", "thread", "created_at"]
    search_fields = ["thread__contact_name"]
    list_filter = ["created_at"]


@admin.register(Summary)
class SummaryAdmin(ModelAdmin):
    list_display = ["id", "thread", "created_at"]
    search_fields = ["thread__contact_name"]
    list_filter = ["created_at"]
