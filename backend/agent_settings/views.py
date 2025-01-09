from django.contrib.auth.mixins import LoginRequiredMixin
from django.urls import reverse_lazy
from django.views.generic import (
    CreateView,
    DeleteView,
    ListView,
    UpdateView,
)

from core.mixins import OrganizationPermissionMixin
from .models import AgentSetting, Tool


class AgentSettingUpdateView(
    LoginRequiredMixin, OrganizationPermissionMixin, UpdateView
):
    model = AgentSetting
    template_name = "agent_settings/agent_setting_update.html"
    fields = [
        "core_llm",
        "condense_llm",
        "embeddings",
        "delay",
        "hide_tool_messages",
    ]

    def get_object(self, queryset=None):
        return AgentSetting.objects.get(
            organization=self.organization,
            version=self.request.session.get("selected_version_id"),
        )


class ToolListView(LoginRequiredMixin, OrganizationPermissionMixin, ListView):
    model = Tool
    template_name = "agent_settings/tool_list.html"
    context_object_name = "tool"


class ToolCreateView(LoginRequiredMixin, OrganizationPermissionMixin, CreateView):
    model = Tool
    template_name = "agent_settings/tool_create.html"
    context_object_name = "tool"
    fields = ["name", "description", "url", "parameters"]

    def get_success_url(self):
        return reverse_lazy(
            "agent_settings:tool_list",
            kwargs={"organization_pk": self.agent_setting.organization.pk},  # type: ignore
        )


class ToolDeleteView(LoginRequiredMixin, OrganizationPermissionMixin, DeleteView):
    model = Tool
    template_name = "agent_settings/tool_delete.html"
    context_object_name = "tool"

    def get_success_url(self):
        return reverse_lazy(
            "agent_settings:tool_list",
            kwargs={"organization_pk": self.agent_setting.organization.pk},  # type: ignore
        )


class ToolUpdateView(LoginRequiredMixin, OrganizationPermissionMixin, UpdateView):
    model = Tool
    template_name = "agent_settings/tool_update.html"
    context_object_name = "tool"
    fields = ["name", "description", "url", "parameters"]
