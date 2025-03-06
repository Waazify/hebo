from django.contrib.auth.mixins import LoginRequiredMixin
from django.urls import reverse
from django.views.generic import (
    CreateView,
    ListView,
    UpdateView,
    DeleteView,
)
from django.http import HttpResponse
from django.template.loader import render_to_string
import logging
from django.shortcuts import get_object_or_404
from django.views import View

from core.mixins import OrganizationPermissionMixin
from .models import Agent, Version, VersionSlug

logger = logging.getLogger(__name__)


class AgentListView(LoginRequiredMixin, OrganizationPermissionMixin, ListView):
    model = Agent
    template_name = "versions/agent_list.html"
    context_object_name = "agents"
    ordering = ["-created_at"]


class VersionDeleteView(LoginRequiredMixin, OrganizationPermissionMixin, DeleteView):
    model = Version
    template_name = "versions/version_confirm_delete.html"

    def get_success_url(self):
        return reverse(
            "agent_list",
            kwargs={
                "organization_pk": self.organization.pk,
            },
        )


class VersionListView(LoginRequiredMixin, OrganizationPermissionMixin, ListView):
    model = Version
    template_name = "versions/version_list.html"
    context_object_name = "versions"
    ordering = ["-created_at"]

    def get_queryset(self):
        return super().get_queryset().filter(agent_id=self.kwargs.get("agent_pk"))

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context["agent"] = Agent.objects.get(pk=self.kwargs.get("agent_pk"))
        return context


class AgentInlineUpdateView(
    LoginRequiredMixin, OrganizationPermissionMixin, UpdateView
):
    model = Agent
    fields = ["name"]
    template_name = "versions/agent_row.html"

    def form_valid(self, form):
        self.object = form.save()
        context = {
            "agent": self.object,
            "organization": self.organization,
        }
        return HttpResponse(
            render_to_string(
                self.template_name,
                context,
                request=self.request,
            )
        )

    def form_invalid(self, form):
        context = {
            "agent": self.get_object(),
            "organization": self.organization,
            "form": form,
        }
        return HttpResponse(
            render_to_string(
                self.template_name,
                context,
                request=self.request,
            ),
            status=400,
        )


class AgentInlineCreateView(
    LoginRequiredMixin, OrganizationPermissionMixin, CreateView
):
    model = Agent
    fields = ["name"]
    template_name = "versions/agent_row.html"

    def get(self, request, *args, **kwargs):
        return HttpResponse(status=405)

    def post(self, request, *args, **kwargs):
        # Create a new agent directly without form validation
        agent = Agent.objects.create(organization=self.organization, name="New Agent")
        return HttpResponse(
            render_to_string(
                self.template_name,
                {"agent": agent, "organization": self.organization},
                request=self.request,
            )
        )


class SetActiveVersionView(LoginRequiredMixin, View):
    def get(self, request, organization_pk, agent_pk, version_pk):
        agent = get_object_or_404(Agent, pk=agent_pk, organization_id=organization_pk)
        version = get_object_or_404(Version, pk=version_pk, agent=agent)
        version_slug = get_object_or_404(VersionSlug, slug=f"{agent.slug}:{version.name}")

        # Store in session
        request.session["selected_agent_id"] = agent.pk
        request.session["selected_version_id"] = version.pk
        request.session["selected_version_slug_id"] = version_slug.pk

        # Create redirect URL
        redirect_url = f"{reverse('knowledge_list', kwargs={'organization_pk': organization_pk})}"

        response = HttpResponse(status=200)
        response["HX-Redirect"] = redirect_url
        return response
