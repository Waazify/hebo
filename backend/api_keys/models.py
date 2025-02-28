import secrets

from django.db import models
from django.db.models.signals import post_save
from django.dispatch import receiver
from django.utils.translation import gettext_lazy as _

from core.managers import OrganizationManagerMixin
from hebo_organizations.models import Organization


class APIKeyManager(OrganizationManagerMixin, models.Manager):
    pass


class APIKey(models.Model):
    organization = models.ForeignKey(
        Organization,
        on_delete=models.CASCADE,
        related_name="api_keys",
        help_text=_("Organization this API key belongs to"),
    )
    name = models.CharField(
        max_length=255, help_text=_("Name of the API key for reference")
    )
    key = models.CharField(
        max_length=128, unique=True, help_text=_("The API key value")
    )
    is_active = models.BooleanField(
        default=True, help_text=_("Whether this API key is active")
    )
    created_at = models.DateTimeField(auto_now_add=True)
    last_used_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]
        constraints = [
            models.UniqueConstraint(
                fields=["organization", "name"],
                name="unique_api_key_name_per_organization",
            )
        ]

    @classmethod
    def generate_key(cls):
        """Generate a random API key"""
        return secrets.token_urlsafe(32)

    def save(self, *args, **kwargs):
        if not self.key:
            self.key = self.generate_key()
        super().save(*args, **kwargs)

    objects = APIKeyManager()


@receiver(post_save, sender=Organization)
def create_initial_api_keys(sender, instance, created, **kwargs):
    """
    Signal handler to create a first API key when a new Organization is created.
    """
    if created:
        APIKey.objects.create(
            organization=instance, name=f"{instance.name} API Key"
        )
