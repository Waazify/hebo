from django.shortcuts import redirect
from django.core.exceptions import PermissionDenied
from django.utils.functional import cached_property
from django.shortcuts import get_object_or_404
from hebo_organizations.models import Organization
from versions.models import Version


class OrganizationPermissionMixin:
    """Mixin to handle organization-specific permissions"""

    org_model = Organization
    org_context_name = "organization"
    owner_required = False  # Set to True for owner-only actions
    require_organization = True  # Set to False for views that can work without an organization


    def get_org_model(self):
        return self.org_model

    def get_context_data(self, **kwargs):
        kwargs.update({self.org_context_name: self.organization})
        return super().get_context_data(**kwargs)  # type: ignore

    @cached_property
    def organization(self) -> Organization:
        organization_pk = self.kwargs.get("organization_pk", None)  # type: ignore
        return get_object_or_404(self.get_org_model(), pk=organization_pk)



    def get_organization(self) -> Organization | None:
        """Get organization from URL parameters"""
        try:
            return Organization.objects.get(pk=self.kwargs.get("organization_pk"))  # type: ignore
        except Organization.DoesNotExist:
            return None

    def dispatch(self, request, *args, **kwargs):
        # Get the organization first
        self.organization = self.get_organization()

        # If organization is required but not found, redirect to base settings
        if self.require_organization and not self.organization:
            return redirect("organization_settings_base")

        # If we have an organization, check permissions
        if self.organization:
            # Check if user is a member of the organization
            if not self.organization.is_member(request.user):
                raise PermissionDenied("You are not a member of this organization")

            # Check if owner permission is required
            if self.owner_required and not self.organization.is_owner(request.user):
                raise PermissionDenied("You must be an owner to perform this action")

        return super().dispatch(request, *args, **kwargs)  # type: ignore

    def get_queryset(self):
        # TODO: Find a better way to do this
        if self.model == Organization:  # type: ignore
            return super().get_queryset().filter(pk=self.organization.pk)  # type: ignore
        if self.model == Version:  # type: ignore
            return super().get_queryset().filter(agent__organization=self.organization)  # type: ignore
        # For other models, filter by organization
        return super().get_queryset().filter(organization=self.organization)  # type: ignore
