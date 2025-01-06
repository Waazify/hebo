from django.contrib.auth.mixins import LoginRequiredMixin
from django.urls import reverse_lazy
from django.views.generic import UpdateView, DeleteView, CreateView
from django.shortcuts import redirect
from django.db import transaction
from django.core.exceptions import PermissionDenied

from .models import Organization, OrganizationUser, OrganizationOwner
from core.mixins import OrganizationPermissionMixin


class OrganizationSettingsView(
    LoginRequiredMixin, OrganizationPermissionMixin, UpdateView
):
    model = Organization
    template_name = "organizations/settings.html"
    fields = ["name"]
    context_object_name = "organization"
    require_organization = (
        False  # Allow access without organization for base settings URL
    )

    def get_object(self, queryset=None):
        return self.organization if self.organization else None

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context["has_organization"] = Organization.objects.filter(
            hebo_organization_user__user=self.request.user
        ).exists()
        # Add is_owner flag to context for template use
        if self.organization:
            context["is_owner"] = self.organization.is_owner(self.request.user)
        return context

    def post(self, request, *args, **kwargs):
        # Check if user is owner before allowing updates
        if self.organization and not self.organization.is_owner(request.user):
            raise PermissionDenied(
                "You must be an owner to update organization settings"
            )
        return super().post(request, *args, **kwargs)

    def get_success_url(self):
        return reverse_lazy(
            "organization_settings",
            kwargs={"organization_pk": self.object.pk},  # type: ignore
        )


class OrganizationCreateView(LoginRequiredMixin, CreateView):
    model = Organization
    fields = ["name"]
    template_name = "organizations/settings.html"

    def form_valid(self, form):
        with transaction.atomic():
            organization = form.save()

            org_user = OrganizationUser.objects.create(
                organization=organization, user=self.request.user
            )

            OrganizationOwner.objects.create(
                organization_user=org_user, organization=organization
            )

            self.object = organization

        return redirect(self.get_success_url())

    def get_success_url(self):
        return reverse_lazy(
            "organization_settings", kwargs={"organization_pk": self.object.pk}
        )


class OrganizationDeleteView(
    LoginRequiredMixin, OrganizationPermissionMixin, DeleteView
):
    model = Organization
    template_name = "organizations/confirm_delete.html"
    success_url = reverse_lazy("organization_settings_base")
    owner_required = True  # Only owners can delete organizations

    def get_object(self, queryset=None):
        return self.organization if self.organization else None
