from django.contrib.auth.mixins import LoginRequiredMixin
from django.urls import reverse_lazy
from django.views.generic import (
    CreateView,
    DeleteView,
    ListView,
    UpdateView,
)
from django.contrib import messages
from django.db import models
from django.http import JsonResponse

from core.mixins import OrganizationPermissionMixin
from .models import AgentSetting, Tool, LLMAdapter
from .forms import LLMAdapterForm, AgentSettingForm


class AgentSettingUpdateView(
    LoginRequiredMixin, OrganizationPermissionMixin, UpdateView
):
    model = AgentSetting
    template_name = "agent_settings/agent_setting_update.html"
    form_class = AgentSettingForm

    def get_object(self, queryset=None):
        obj, _ = AgentSetting.objects.get_or_create(
            organization=self.organization,
            version=self.request.session.get("selected_version_id"),
            defaults={
                "delay": False,
                "hide_tool_messages": False,
            },
        )
        return obj

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)

        # Get both organization-specific and default adapters
        adapters = LLMAdapter.objects.filter(
            models.Q(organization=self.organization) | models.Q(is_default=True)
        )

        # Filter adapters by type and add to context
        chat_adapters = adapters.filter(model_type=LLMAdapter.ModelType.CHAT).order_by(
            "-is_default", "provider", "name"
        )

        embedding_adapters = adapters.filter(
            model_type=LLMAdapter.ModelType.EMBEDDING
        ).order_by("-is_default", "provider", "name")

        context.update(
            {
                "chat_adapters": chat_adapters,
                "embedding_adapters": embedding_adapters,
                "agent_setting": self.get_object(),  # Ensure we have the latest object
            }
        )

        return context

    def form_valid(self, form):
        response = super().form_valid(form)
        return response

    def get_success_url(self):
        return reverse_lazy(
            "agent_setting_update",
            kwargs={"organization_pk": self.organization.pk},
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
            "tool_list",
            kwargs={"organization_pk": self.agent_setting.organization.pk},  # type: ignore
        )


class ToolDeleteView(LoginRequiredMixin, OrganizationPermissionMixin, DeleteView):
    model = Tool
    template_name = "agent_settings/tool_delete.html"
    context_object_name = "tool"

    def get_success_url(self):
        return reverse_lazy(
            "tool_list",
            kwargs={"organization_pk": self.agent_setting.organization.pk},  # type: ignore
        )


class ToolUpdateView(LoginRequiredMixin, OrganizationPermissionMixin, UpdateView):
    model = Tool
    template_name = "agent_settings/tool_update.html"
    context_object_name = "tool"
    fields = ["name", "description", "url", "parameters"]


class LLMAdapterListView(LoginRequiredMixin, OrganizationPermissionMixin, ListView):
    model = LLMAdapter
    template_name = "agent_settings/llm_adapter_list.html"
    context_object_name = "adapters"

    def get_queryset(self):
        # Get both organization-specific and default adapters
        return LLMAdapter.objects.filter(
            models.Q(organization=self.organization) | models.Q(is_default=True)
        ).order_by("-is_default", "provider", "name")


class LLMAdapterCreateView(LoginRequiredMixin, OrganizationPermissionMixin, CreateView):
    model = LLMAdapter
    form_class = LLMAdapterForm

    def get_form_kwargs(self):
        kwargs = super().get_form_kwargs()
        kwargs["organization"] = self.organization
        return kwargs

    def form_valid(self, form):
        try:
            self.object = form.save()
            return JsonResponse(
                {
                    "status": "success",
                    "adapter": {
                        "id": self.object.id,
                        "name": self.object.name,
                        "is_default": self.object.is_default,
                    },
                }
            )
        except Exception as e:
            return JsonResponse(
                {
                    "status": "error",
                    "message": str(e),
                },
                status=400,
            )


class LLMAdapterDeleteView(LoginRequiredMixin, OrganizationPermissionMixin, DeleteView):
    model = LLMAdapter
    template_name = "agent_settings/llm_adapter_confirm_delete.html"

    def get_queryset(self):
        # Only allow deleting organization's own adapters
        return LLMAdapter.objects.filter(organization=self.organization)

    def delete(self, request, *args, **kwargs):
        response = super().delete(request, *args, **kwargs)
        messages.success(self.request, "LLM adapter deleted successfully.")
        return response

    def get_success_url(self):
        return reverse_lazy("llm_adapter_list")
