from django.db import models
from django.dispatch import receiver
from django.utils.translation import gettext_lazy as _
from django.core.exceptions import ValidationError

from core.managers import OrganizationManagerMixin
from hebo_organizations.models import Organization
from versions.models import Version, initial_version_created


class AgentSettingManager(OrganizationManagerMixin, models.Manager):
    pass


class LLMAdapter(models.Model):
    class ModelType(models.TextChoices):
        CHAT = "chat", "Chat"
        EMBEDDING = "embedding", "Embedding"

    class ProviderType(models.TextChoices):
        ANTHROPIC = "anthropic", "Anthropic"
        OPENAI = "openai", "OpenAI"
        AZURE = "azure", "Azure"
        BEDROCK = "bedrock", "AWS Bedrock"
        VERTEX = "vertex-ai", "Vertex AI"
        VOYAGE = "voyage", "Voyage"

    is_default = models.BooleanField(
        default=False,
        help_text=_("If True, this adapter is available to all organizations"),
    )
    organization = models.ForeignKey(
        Organization,
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name="llm_adapters",
        help_text=_("Organization this adapter belongs to (null if default)"),
    )
    model_type = models.CharField(
        max_length=20,
        choices=ModelType.choices,
        help_text=_("Type of model (chat, embedding, or vision)"),
        default=ModelType.CHAT,
    )
    provider = models.CharField(
        max_length=20,
        choices=ProviderType.choices,
        help_text=_("AI provider for this model"),
    )
    api_base = models.URLField(
        blank=True,
        help_text=_("Base URL for API calls (if different from provider default)"),
    )
    name = models.CharField(
        max_length=100,
        help_text=_("Name/ID of the model from the provider"),
    )
    aws_region = models.CharField(
        max_length=50,
        blank=True,
        help_text=_("AWS region (required for Bedrock)"),
    )
    api_key = models.CharField(
        max_length=2000,
        blank=True,
        help_text=_("API key, service account JSON, or other authentication credentials"),
    )
    aws_access_key_id = models.CharField(
        max_length=200,
        blank=True,
        help_text=_("AWS Access Key ID (required for Bedrock)"),
    )
    aws_secret_access_key = models.CharField(
        max_length=200,
        blank=True,
        help_text=_("AWS Secret Access Key (required for Bedrock)"),
    )

    class Meta:
        indexes = [
            models.Index(fields=["organization", "model_type"]),
            models.Index(fields=["is_default", "model_type"]),
        ]
        constraints = [
            models.CheckConstraint(
                check=(
                    models.Q(is_default=True, organization__isnull=True) |
                    models.Q(is_default=False, organization__isnull=False)
                ),
                name="organization_xor_default"
            )
        ]

    def clean(self):
        if self.is_default and self.organization:
            raise ValidationError(
                _("Default adapters cannot be associated with an organization")
            )
        if not self.is_default and not self.organization:
            raise ValidationError(
                _("Custom adapters must be associated with an organization")
            )
        if self.provider == self.ProviderType.BEDROCK and (
            not self.aws_region
            or not self.aws_access_key_id
            or not self.aws_secret_access_key
        ):
            raise ValidationError(
                _(
                    "AWS region, access key ID, and secret access key are required for Bedrock provider"
                )
            )
        if self.provider != self.ProviderType.BEDROCK and not self.api_key:
            raise ValidationError(
                _("API key is required for non-Bedrock providers")
            )

    def save(self, *args, **kwargs):
        self.clean()
        super().save(*args, **kwargs)

    def __str__(self):
        return f"{self.name} ({self.get_provider_display()} - {'Default' if self.is_default else 'Custom'})"  # type: ignore


class AgentSetting(models.Model):
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
    core_llm = models.ForeignKey(
        LLMAdapter,
        on_delete=models.PROTECT,
        null=True,
        blank=True,
        related_name="core_llm_settings",
        limit_choices_to={"model_type": LLMAdapter.ModelType.CHAT},
    )
    condense_llm = models.ForeignKey(
        LLMAdapter,
        on_delete=models.PROTECT,
        null=True,
        blank=True,
        related_name="condense_llm_settings",
        limit_choices_to={"model_type": LLMAdapter.ModelType.CHAT},
    )
    embeddings = models.ForeignKey(
        LLMAdapter,
        on_delete=models.PROTECT,
        null=True,
        blank=True,
        related_name="embedding_settings",
        limit_choices_to={"model_type": LLMAdapter.ModelType.EMBEDDING},
    )
    vision_llm = models.ForeignKey(
        LLMAdapter,
        on_delete=models.PROTECT,
        null=True,
        blank=True,
        related_name="vision_llm_settings",
        limit_choices_to={"model_type": LLMAdapter.ModelType.CHAT},
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
        super().clean()
        # Validate that the organization has access to the selected adapters
        for field in [self.core_llm, self.condense_llm, self.embeddings, self.vision_llm]:
            if field and not field.is_default and field.organization != self.organization:
                raise ValidationError(
                    _(f"Selected adapter {field.name} is not available for this organization")
                )

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
