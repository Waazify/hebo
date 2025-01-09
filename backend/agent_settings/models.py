from django.db import models
from django.dispatch import receiver
from django.utils.translation import gettext_lazy as _

from core.managers import OrganizationManagerMixin
from hebo_organizations.models import Organization
from versions.models import Version, initial_version_created


class AgentSettingManager(OrganizationManagerMixin, models.Manager):
    pass


class AgentSetting(models.Model):
    class LLMChoices(models.TextChoices):
        GPT4_O = "gpt4o", "GPT-4o"
        GPT4_O_MINI = "gpt4o_mini", "GPT-4o mini"
        O1 = "o1", "o1"
        O1_MINI = "o1_mini", "o1 mini"
        CLAUDE_3_5_SONNET = "claude_3_5_sonnet", "Claude-3.5-Sonnet"
        CLAUDE_3_5_HAIKU = "claude_3_5_haiku", "Claude-3.5-Haiku"

    class EmbeddingChoices(models.TextChoices):
        ADA_002 = "ada002", "text-embedding-ada-002"
        MINILM = "minilm", "all-MiniLM-L6-v2"
        MPNet = "mpnet", "all-mpnet-base-v2"
        BGER = "bger", "bge-large-en"

    organization = models.ForeignKey(
        Organization,
        on_delete=models.CASCADE,
        related_name="agent_settings",
        help_text=_("Organization this agent belongs to"),
    )

    version = models.OneToOneField(
        Version,
        on_delete=models.CASCADE,
        related_name="agent_settings",
        help_text=_("The version these settings belong to"),
        unique=True,
    )

    core_llm = models.CharField(
        max_length=20, choices=LLMChoices.choices, default=LLMChoices.GPT4_O
    )
    condense_llm = models.CharField(
        max_length=20, choices=LLMChoices.choices, default=LLMChoices.GPT4_O_MINI
    )
    embeddings = models.CharField(
        max_length=20,
        choices=EmbeddingChoices.choices,
        default=EmbeddingChoices.ADA_002,
    )
    delay = models.BooleanField(default=False)
    hide_tool_messages = models.BooleanField(default=False)

    objects = AgentSettingManager()

    class Meta:
        indexes = [
            models.Index(fields=["version"]),
        ]
        constraints = [
            models.UniqueConstraint(fields=["version"], name="unique_version_settings")
        ]

    def clean(self):
        from django.core.exceptions import ValidationError

        if not self.pk:
            existing = AgentSetting.objects.filter(version=self.version).exists()
            if existing:
                raise ValidationError(
                    {"version": _("An AgentSetting already exists for this version.")}
                )

        super().clean()

    def save(self, *args, **kwargs):
        self.clean()
        super().save(*args, **kwargs)


class ToolManager(OrganizationManagerMixin, models.Manager):
    def get_queryset(self):
        return super().get_queryset().select_related("agent_setting__version__agent")


class Tool(models.Model):
    class ToolType(models.TextChoices):
        ACTION = "action", "Action"
        DATA_SOURCE = "data_source", "Data Source"

    agent_setting = models.ForeignKey(
        AgentSetting,
        on_delete=models.CASCADE,
        related_name="tools",
        help_text="The agent setting this tool belongs to",
    )
    name = models.CharField(max_length=200)
    description = models.TextField(help_text="Description of what the tool does")
    output_template = models.TextField(help_text="Template to format the tool's output")
    tool_type = models.CharField(
        max_length=20, choices=ToolType.choices, default=ToolType.ACTION
    )

    # Action-specific fields
    openapi_url = models.URLField(
        blank=True, null=True, help_text="OpenAPI URL for Action type tools"
    )
    auth_token = models.CharField(
        max_length=255,
        blank=True,
        null=True,
        help_text="Optional authentication token for API",
    )

    # Data Source-specific fields
    db_connection_string = models.CharField(
        max_length=255,
        blank=True,
        null=True,
        help_text="Database connection string for Data Source type tools",
    )
    query = models.TextField(
        blank=True,
        null=True,
        help_text="Query to execute for Data Source type tools",
    )

    objects = ToolManager()

    class Meta:
        indexes = [
            models.Index(fields=["agent_setting"]),
        ]

    def clean(self):
        from django.core.exceptions import ValidationError

        if self.tool_type == self.ToolType.ACTION and not self.openapi_url:
            raise ValidationError(
                {"openapi_url": "OpenAPI URL is required for Action type tools"}
            )

        if self.tool_type == self.ToolType.DATA_SOURCE:
            if not self.db_connection_string:
                raise ValidationError(
                    {
                        "db_connection_string": "Database connection string is required for Data Source type tools"
                    }
                )
            if not self.query:
                raise ValidationError(
                    {"query": "Query is required for Data Source type tools"}
                )

    def save(self, *args, **kwargs):
        self.clean()
        super().save(*args, **kwargs)


@receiver(initial_version_created)
def create_default_agent_setting(sender, created, agent, version, **kwargs):
    """
    Signal handler to create a default agent setting when a new agent is created.
    """
    if created:
        AgentSetting.objects.create(
            organization=agent.organization, version=version
        )
