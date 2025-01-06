from django.dispatch import receiver
from django.contrib.auth import get_user_model
from allauth.account.signals import user_signed_up
from .models import Organization, OrganizationUser, OrganizationOwner

User = get_user_model()


@receiver(user_signed_up)
def create_default_organization(sender, request, user, **kwargs):
    # Only create organization if user doesn't have one
    if not OrganizationUser.objects.filter(user=user).exists():
        # Create organization name using first_name or email
        org_name = (
            f"{user.first_name}'s Organization"
            if user.first_name
            else f"{user.email.split('@')[0]}'s Organization"
        )

        # Create the organization
        org = Organization.objects.create(name=org_name)

        # Create organization user
        org_user = OrganizationUser.objects.create(
            organization=org, user=user, is_admin=True
        )

        # Create organization owner
        OrganizationOwner.objects.create(organization=org, organization_user=org_user)
