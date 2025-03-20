import logging
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
from django.http import JsonResponse, HttpResponseNotAllowed
from rest_framework import viewsets
from rest_framework.exceptions import ValidationError

from core.authentication import APIKeyAuthentication
from core.mixins import OrganizationPermissionMixin
from versions.models import VersionSlug
from .forms import LLMAdapterForm, AgentSettingForm
from .models import AgentSetting, Tool, LLMAdapter
from .serializers import AgentSettingSerializer, ToolSerializer

# Configure a logger for this module
logger = logging.getLogger(__name__)


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
                "include_last_24h_history": False,
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
    http_method_names = ["post"]

    def get(self, request, *args, **kwargs):
        # Log an error if a GET request is made
        logger.error(
            "GET request received on LLMAdapterCreateView; only POST is allowed."
        )
        return HttpResponseNotAllowed(["POST"])

    def get_form_kwargs(self):
        kwargs = super().get_form_kwargs()
        logger.debug(
            "LLMAdapterCreateView.get_form_kwargs: Setting organization: %s",
            self.organization,
        )
        kwargs["organization"] = self.organization
        return kwargs

    def form_valid(self, form):
        try:
            self.object = form.save()
            logger.info(
                "LLMAdapterCreateView.form_valid: Successfully created adapter: %s",
                self.object,
            )
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
            logger.exception(
                "LLMAdapterCreateView.form_valid: Exception while saving form: %s", e
            )
            return JsonResponse(
                {
                    "status": "error",
                    "message": str(e),
                },
                status=400,
            )

    def form_invalid(self, form):
        # Log the form errors to help debug why validation failed
        logger.error("LLMAdapterCreateView.form_invalid: Form errors: %s", form.errors)
        return JsonResponse(
            {"status": "error", "errors": form.errors},
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


class AgentSettingViewSet(viewsets.ModelViewSet):
    """
    API endpoint for accessing agent settings.
    Requires X-API-Key authentication and supports filtering by agent_slug.
    """

    serializer_class = AgentSettingSerializer
    authentication_classes = [APIKeyAuthentication]
    permission_classes = []

    def get_queryset(self):
        # Get organization from authentication
        organization = self.request.auth  # type: ignore

        # Get agent_slug from query params
        agent_version = self.request.query_params.get("agent_version")  # type: ignore
        if not agent_version:
            raise ValidationError("agent_version query parameter is required")

        # Get version from agent_slug
        try:
            version_slug = VersionSlug.objects.get(slug=agent_version)
            version = version_slug.version
        except VersionSlug.DoesNotExist:
            raise ValidationError(
                f"No version found for agent_version: {agent_version}"
            )

        # Filter agent settings by organization and version
        return AgentSetting.objects.filter(organization=organization, version=version)

    def list(self, request, *args, **kwargs):
        """
        List all agent settings for the authenticated organization and specified agent_slug.
        """
        return super().list(request, *args, **kwargs)


class ToolViewSet(viewsets.ModelViewSet):
    """
    API endpoint for accessing tools.
    """

    serializer_class = ToolSerializer
    authentication_classes = [APIKeyAuthentication]
    permission_classes = []

    def get_queryset(self):
        # Get organization from authentication
        organization = self.request.auth  # type: ignore

        # Get agent_slug from query params
        agent_version = self.request.query_params.get("agent_version")  # type: ignore
        if not agent_version:
            raise ValidationError("agent_version query parameter is required")

        # Get version from agent_slug
        try:
            version_slug = VersionSlug.objects.get(slug=agent_version)
            version = version_slug.version
        except VersionSlug.DoesNotExist:
            raise ValidationError(
                f"No version found for agent_version: {agent_version}"
            )

        # Filter tools by agent_setting that matches the organization and version
        return Tool.objects.filter(
            agent_setting__organization=organization,
            agent_setting__version=version
        )

    def list(self, request, *args, **kwargs):
        """
        List all tools for the authenticated organization and specified agent_slug.
        """
        return super().list(request, *args, **kwargs)
