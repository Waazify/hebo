from django.db import models
from hebo_organizations.models import OrganizationUser


class OrganizationManagerMixin(models.Manager):
    def get_queryset(self):
        return super().get_queryset()

    def for_organization(self, organization):
        return self.get_queryset().filter(organization=organization)

    def for_user(self, user):
        # Get all organizations the user belongs to
        user_organizations = OrganizationUser.objects.filter(user=user).values_list(
            "organization", flat=True
        )

        return self.get_queryset().filter(organization__in=user_organizations)
