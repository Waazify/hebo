import markdown
from typing import TYPE_CHECKING

from django.core.exceptions import ValidationError
from django.db import models
from django.utils import timezone
from django.utils.safestring import mark_safe
from django.utils.translation import gettext_lazy as _
from pgvector.django import VectorField

from agent_settings.models import AgentSetting
from hebo_organizations.models import Organization
from core.managers import OrganizationManagerMixin

if TYPE_CHECKING:
    from django.db.models.manager import RelatedManager


class PageManager(OrganizationManagerMixin, models.Manager):
    pass


class Page(models.Model):
    """
    Model for storing markdown-formatted pages.

    The content field stores markdown text that can be rendered to HTML.
    Supports standard markdown syntax including:
    - Headers (# ## ###)
    - Lists (* - +)
    - Code blocks (``` ```)
    - Links [text](url)
    - Images ![alt](url)
    """

    organization = models.ForeignKey(
        Organization,
        on_delete=models.CASCADE,
        related_name="pages",
        help_text=_("Organization this agent belongs to"),
    )
    version = models.ForeignKey(
        "versions.Version",
        on_delete=models.CASCADE,
        related_name="pages",
        help_text=_("The version this page belongs to"),
    )

    parts: "RelatedManager[Part]"  # Added for type hinting

    title = models.CharField(max_length=200)
    content = models.TextField(
        help_text="Content in markdown format. Supports standard markdown syntax."
    )

    # Metadata fields
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)
    is_published = models.BooleanField(default=False)

    # Optional: Add parent-child relationship for nested pages
    parent = models.ForeignKey(
        "self",
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="children",
    )

    objects = PageManager()

    class Meta:
        ordering = ["-updated_at"]
        verbose_name = "Page"
        verbose_name_plural = "Pages"

    def save(self, *args, **kwargs):
        # Check if this is an update and content changed
        if self.pk:
            old_page = Page.objects.get(pk=self.pk)
            if old_page.content != self.content:
                # Mark all existing parts as invalid
                self.parts.update(is_valid=False)

        super().save(*args, **kwargs)

        # Generate new parts
        self.generate_parts()

    def generate_parts(self):
        """
        Dummy function that generates parts based on double newlines.
        In reality, you'd have more sophisticated parsing logic.
        """
        # Split content by double newlines
        sections = self.content.split("\n\n")
        current_line = 0

        for i, section in enumerate(sections):
            if not section.strip():  # Skip empty sections
                current_line += section.count("\n") + 2
                continue

            # Calculate line numbers
            section_lines = section.count("\n") + 1
            end_line = current_line + section_lines

            # Generate content hash
            content_hash = Part._generate_hash(section)

            # Check if a valid part already exists for this section
            existing_part = self.parts.filter(
                start_line=current_line,
                end_line=end_line,
                content_hash=content_hash,
                is_valid=True,
            ).first()

            if not existing_part:
                # Either create new part or update invalid one
                _, _ = self.parts.update_or_create(
                    identifier=f"section_{i}",  # Simple identifier for demo
                    defaults={
                        "start_line": current_line,
                        "end_line": end_line,
                        "content_hash": content_hash,
                        "content_type": ContentType.BEHAVIOUR,  # Dummy assignment
                        "is_valid": True,
                    },
                )

            current_line = end_line + 2  # +2 for the double newline separator

    def get_html_content(self):
        """
        Renders the markdown content to safe HTML.

        Returns:
            SafeString: HTML-rendered content, safe for template rendering
        """
        md = markdown.Markdown(
            extensions=[
                "extra",  # Tables, footnotes, attribute lists, etc.
                "codehilite",  # Code block syntax highlighting
                "fenced_code",  # Fenced code blocks
                "toc",  # Table of contents
                "tables",  # Tables support
            ]
        )
        return mark_safe(md.convert(self.content))

    @property
    def word_count(self):
        """
        Returns the approximate word count of the markdown content.

        Returns:
            int: Number of words in the content
        """
        return len(self.content.split())


class ContentType(models.TextChoices):
    """Available part types for content sections."""

    BEHAVIOUR = "behaviour", "Behaviour"
    SCENARIO = "scenario", "Scenario"
    EXAMPLE = "example", "Example"


class PartManager(OrganizationManagerMixin, models.Manager):
    def get_queryset(self):
        return super().get_queryset().select_related("page__organization")


class Part(models.Model):
    """
    Represents an automatically extracted section from a page's content.
    Parts are identified through specific markdown formatting and processed
    via background tasks. Each part maintains a reference to its original
    content location and tracks content validity.
    """

    page = models.ForeignKey("Page", on_delete=models.CASCADE, related_name="parts")

    # Content reference
    start_line = models.PositiveIntegerField(
        help_text="Starting line number in the original content"
    )
    end_line = models.PositiveIntegerField(
        help_text="Ending line number in the original content"
    )
    content_hash = models.CharField(
        max_length=64,
        help_text="Hash of the original content section for tracking changes",
    )

    # Part metadata
    content_type = models.CharField(
        max_length=20, choices=ContentType.choices, db_index=True
    )
    identifier = models.CharField(
        max_length=100, help_text="Unique identifier for this part within its page"
    )
    is_handover = models.BooleanField(
        default=False,
        help_text="Special tag for downstream processing (valid only for scenarios and examples)",
    )

    # Tracking
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)
    is_valid = models.BooleanField(
        default=True,
        help_text="Indicates if the original content has changed and needs reprocessing",
    )

    objects = PartManager()

    class Meta:
        ordering = ["start_line"]
        constraints = [
            models.UniqueConstraint(
                fields=["page", "identifier"], name="unique_part_identifier_per_page"
            )
        ]
        indexes = [
            models.Index(fields=["page", "content_type", "is_valid"]),
        ]

    def clean(self):
        if self.is_handover and self.content_type == ContentType.BEHAVIOUR:
            raise ValidationError(
                {
                    "is_handover": "Handover tag can only be applied to scenarios and examples"
                }
            )
        if self.end_line <= self.start_line:
            raise ValidationError(
                {"end_line": "End line must be greater than start line"}
            )

    def save(self, *args, **kwargs):
        self.full_clean()
        super().save(*args, **kwargs)

    def get_content(self):
        """
        Retrieves the current content for this part from the page.
        Also verifies content validity by comparing hashes.
        """
        lines = self.page.content.splitlines()
        if self.start_line >= len(lines) or self.end_line > len(lines):
            self.is_valid = False
            self.save(update_fields=["is_valid"])
            return None

        content = "\n".join(lines[self.start_line : self.end_line])
        if self.content_hash != self._generate_hash(content):
            self.is_valid = False
            self.save(update_fields=["is_valid"])
        return content

    @staticmethod
    def _generate_hash(content: str) -> str:
        """Generates a hash for the given content."""
        from hashlib import sha256

        return sha256(content.encode()).hexdigest()

    def __str__(self):
        return f"{self.content_type}: {self.identifier} ({self.page.title})"


def cleanup_invalid_parts():
    """
    Utility function to handle invalid parts.
    Could be called periodically or manually as needed.
    """
    # Delete parts that have been invalid for more than 24 hours
    cutoff = timezone.now() - timezone.timedelta(hours=24)
    Part.objects.filter(is_valid=False, updated_at__lt=cutoff).delete()


class VectorManager(OrganizationManagerMixin, models.Manager):
    def get_queryset(self):
        return super().get_queryset().select_related("part__page__organization")


class VectorStore(models.Model):
    """
    Stores vector embeddings for Page Parts using pgvector.
    The vector dimensions vary by embedding model:
    - ada002: 1536 dimensions
    - minilm: 384 dimensions
    - mpnet: 768 dimensions
    - bge-large-en: 1024 dimensions
    """

    class EmbeddingModel(models.TextChoices):
        ADA_002 = "ada002", "text-embedding-ada-002"  # 1536 dims
        MINILM = "minilm", "all-MiniLM-L6-v2"  # 384 dims
        MPNET = "mpnet", "all-mpnet-base-v2"  # 768 dims
        BGER = "bger", "bge-large-en"  # 1024 dims

    DIMENSION_MAP = {
        EmbeddingModel.ADA_002: 1536,
        EmbeddingModel.MINILM: 384,
        EmbeddingModel.MPNET: 768,
        EmbeddingModel.BGER: 1024,
    }

    part = models.ForeignKey("Part", on_delete=models.CASCADE, related_name="vectors")

    embedding_model = models.CharField(
        max_length=20,
        choices=EmbeddingModel.choices,
        help_text="Must match the embedding model in agent settings",
    )

    vector = VectorField(dimensions=1536)  # Max dimensions (ada-002)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    metadata = models.JSONField(default=dict)
    objects = VectorManager()

    class Meta:
        indexes = [
            models.Index(fields=["id"]),
        ]
        constraints = [
            models.UniqueConstraint(
                fields=["part", "embedding_model"],
                name="unique_vector_per_part",
            )
        ]

    def clean(self):
        try:
            agent_settings = AgentSetting.objects.get(version=self.part.page.version)
        except AgentSetting.DoesNotExist:
            raise ValidationError(
                "Cannot validate embedding model - no agent settings found"
            )

        # Convert string to enum value before validation
        embedding_model = self.EmbeddingModel(self.embedding_model)
        if embedding_model.value != agent_settings.embeddings:
            raise ValidationError(
                {
                    "embedding_model": f"Embedding model must match agent settings ({agent_settings.embeddings})"
                }
            )

        # Use enum value to access DIMENSION_MAP
        expected_dims = self.DIMENSION_MAP[embedding_model]
        if self.vector is not None and len(self.vector) != expected_dims:
            raise ValidationError(
                {
                    "vector": f"Vector must have {expected_dims} dimensions for {self.embedding_model}"
                }
            )

    def save(self, *args, **kwargs):
        self.full_clean()
        super().save(*args, **kwargs)

    def __str__(self):
        return f"Vector for {self.part} using {self.embedding_model}"
