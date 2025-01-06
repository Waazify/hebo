from django.contrib import admin
from django.contrib.admin import ModelAdmin
from .models import Organization, OrganizationUser, OrganizationOwner, OrganizationInvitation

# Unregister third-party organization models
from organizations.models import (
    Organization as BaseOrganization,
    OrganizationUser as BaseOrganizationUser,
    OrganizationOwner as BaseOrganizationOwner,
    OrganizationInvitation as BaseOrganizationInvitation,
)

# Try to unregister third-party models if they are registered
for model in [BaseOrganization, BaseOrganizationUser, BaseOrganizationOwner, BaseOrganizationInvitation]:
    try:
        admin.site.unregister(model)
    except admin.sites.NotRegistered:
        pass

@admin.register(Organization)
class OrganizationAdmin(ModelAdmin):
    list_display = ['name', 'slug', 'created', 'modified', 'stripe_customer']
    search_fields = ['name', 'slug']
    list_filter = ['created', 'modified']

@admin.register(OrganizationUser)
class OrganizationUserAdmin(ModelAdmin):
    list_display = ['user', 'organization', 'created']
    search_fields = ['user__email', 'organization__name']
    list_filter = ['created']

@admin.register(OrganizationOwner)
class OrganizationOwnerAdmin(ModelAdmin):
    list_display = ['organization', 'organization_user']
    search_fields = ['organization__name', 'organization_user__user__email']

@admin.register(OrganizationInvitation)
class OrganizationInvitationAdmin(ModelAdmin):
    list_display = ['invitee_identifier', 'organization', 'created']
    search_fields = ['invitee_identifier', 'organization__name']
    list_filter = ['created']
