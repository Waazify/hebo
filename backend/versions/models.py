from django.db import models
from django.core.exceptions import ValidationError
from django.utils.translation import gettext_lazy as _
from django.db.models.signals import post_save, post_delete
from django.dispatch import receiver
from hebo_organizations.models import Organization
import uuid
from django.urls import reverse
from core.managers import OrganizationManagerMixin


class AgentManager(OrganizationManagerMixin, models.Manager):
    pass


class Agent(models.Model):
    organization = models.ForeignKey(
        Organization,
        on_delete=models.CASCADE,
        related_name="agents",
        help_text=_("Organization this agent belongs to"),
    )
    name = models.CharField(
        max_length=255,
        help_text=_("Name of the agent"),
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    objects = AgentManager()

    def __str__(self):
        return self.name

    class Meta:
        ordering = ["-created_at"]
        constraints = [
            models.UniqueConstraint(
                fields=["organization", "name", "created_at"],
                name="unique_agent_name_timestamp_per_organization",
            )
        ]


class VersionManager(OrganizationManagerMixin, models.Manager):
    def get_queryset(self):
        return super().get_queryset().select_related("agent__organization")

    def current(self):
        return self.filter(status='CURRENT').first()


class Version(models.Model):
    class Status(models.TextChoices):
        CURRENT = "current", _("Current")
        NEXT = "next", _("Next")
        PAST = "past", _("Past")

    agent = models.ForeignKey(
        Agent,
        on_delete=models.CASCADE,
        related_name="versions",
        help_text=_("The agent this version belongs to"),
    )
    name = models.CharField(
        max_length=255,
        help_text=_("Version name (e.g. v1, v2, etc.)"),
    )
    status = models.CharField(
        max_length=20,
        choices=Status.choices,
        default=Status.CURRENT,
        help_text=_("Status of this version"),
    )
    url_id = models.UUIDField(
        default=uuid.uuid4,
        editable=False,
        unique=True,
        help_text=_("Unique identifier for public access URL"),
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    objects = VersionManager()

    class Meta:
        ordering = ["-created_at"]
        constraints = [
            models.UniqueConstraint(
                fields=["agent", "name"], name="unique_version_name_per_agent"
            ),
            models.UniqueConstraint(
                fields=["agent", "status"],
                condition=models.Q(status="current"),
                name="unique_current_version_per_agent",
            ),
            models.UniqueConstraint(
                fields=["agent", "status"],
                condition=models.Q(status="next"),
                name="unique_next_version_per_agent",
            ),
        ]

    def __str__(self):
        return f"{self.agent.name} - {self.name}"

    def get_public_url(self):
        """
        Returns the public URL for this version
        """
        return reverse("version-public-chat", kwargs={"url_id": self.url_id})

    def get_absolute_url(self):
        """
        Returns the authenticated URL for this version
        """
        return reverse("version-detail", kwargs={"pk": self.pk})

    def clean(self):
        if self.status == self.Status.CURRENT:
            # Check if there's already a current version for this agent
            current_version = (
                Version.objects.filter(agent=self.agent, status=self.Status.CURRENT)
                .exclude(pk=self.pk)
                .exists()
            )

            if current_version:
                raise ValidationError(
                    _("There can only be one current version per agent.")
                )

        elif self.status == self.Status.NEXT:
            # Check if there's already a next version for this agent
            next_version = (
                Version.objects.filter(agent=self.agent, status=self.Status.NEXT)
                .exclude(pk=self.pk)
                .exists()
            )

            if next_version:
                raise ValidationError(
                    _("There can only be one next version per agent.")
                )

    def save(self, *args, **kwargs):
        self.clean()
        super().save(*args, **kwargs)


@receiver(post_save, sender=Agent)
def create_initial_version(sender, instance, created, **kwargs):
    """
    Signal handler to create Version 1 when a new Agent is created.
    """
    if created:
        Version.objects.create(agent=instance, name="v1", status=Version.Status.NEXT)


@receiver(post_save, sender=Organization)
def create_initial_agent(sender, instance, created, **kwargs):
    """
    Signal handler to create Agent #1 when a new Organization is created.
    """
    if created:
        Agent.objects.create(
            organization=instance,
            name="Your First Agent"
        )


@receiver(post_delete, sender=Version)
def delete_empty_agent(sender, instance, **kwargs):
    """
    Signal handler to delete an agent when its last version is deleted.
    """
    # Check if this was the last version
    if not Version.objects.filter(agent=instance.agent).exists():
        instance.agent.delete()
