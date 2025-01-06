from django.db import models
import uuid
from organizations.abstract import (
    AbstractOrganization,
    AbstractOrganizationUser,
    AbstractOrganizationOwner,
    AbstractOrganizationInvitation,
)
from djstripe.models import Customer


def generate_org_id():
    return f"org-{uuid.uuid4()}"


class Organization(AbstractOrganization):
    id = models.CharField(
        primary_key=True,
        default=generate_org_id,
        max_length=40,
        editable=False,
    )
    stripe_customer = models.OneToOneField(
        Customer,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="hebo_organization",
    )

    class Meta(AbstractOrganization.Meta):
        abstract = False


class OrganizationUser(AbstractOrganizationUser):
    organization = models.ForeignKey(
        Organization, on_delete=models.CASCADE, related_name="hebo_organization_user"
    )
    user = models.ForeignKey(
        "auth.User", on_delete=models.CASCADE, related_name="hebo_organization_user"
    )

    class Meta(AbstractOrganizationUser.Meta):
        abstract = False


class OrganizationOwner(AbstractOrganizationOwner):
    organization_user = models.OneToOneField(
        OrganizationUser,
        on_delete=models.CASCADE,
        related_name="hebo_organization_owner",
    )

    class Meta(AbstractOrganizationOwner.Meta):
        abstract = False


class OrganizationInvitation(AbstractOrganizationInvitation):
    organization = models.ForeignKey(
        Organization,
        on_delete=models.CASCADE,
        related_name="hebo_organization_invitation",
    )

    class Meta(AbstractOrganizationInvitation.Meta):
        abstract = False
