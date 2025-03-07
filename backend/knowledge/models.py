import logging
import re
import markdown
from hashlib import sha256
from typing import TYPE_CHECKING

from django.core.exceptions import ValidationError
from django.db import models, IntegrityError
from django.utils import timezone
from django.utils.safestring import mark_safe
from django.utils.translation import gettext_lazy as _
from pgvector.django import VectorField

from agent_settings.models import AgentSetting
from core.managers import OrganizationManagerMixin
from hebo_organizations.models import Organization

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from django.db.models.manager import RelatedManager


class PartGenerationError(Exception):
    """Custom exception for part generation errors."""
    pass


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

    # Add position field for ordering
    position = models.PositiveIntegerField(
        default=0,
        db_index=True,
        help_text=_("Position of the page within its parent level"),
    )

    # Existing parent field
    parent = models.ForeignKey(
        "self",
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="children",
    )

    # Add a new attribute to control part generation
    _skip_part_generation = False

    objects = PageManager()

    class Meta:
        ordering = ["parent__id", "position", "-updated_at"]
        verbose_name = "Page"
        verbose_name_plural = "Pages"

    def save(self, *args, **kwargs):
        # Check if the version status allows editing
        skip_version_check = kwargs.pop("skip_version_check", False)

        if not skip_version_check:
            # Only allow edits if version status is 'next'
            if self.version.status != "next":
                raise IntegrityError(
                    f"Cannot modify page for version with status '{self.version.status}'. "
                    f"Only pages in versions with 'next' status can be modified."
                )

        if not self.position and self.parent:
            # If no position specified, put it at the end of its parent's children
            last_sibling = (
                Page.objects.filter(parent=self.parent).order_by("-position").first()
            )
            self.position = (last_sibling.position + 1) if last_sibling else 0
        elif not self.position:
            # If no parent, put it at the end of root level pages
            last_root = (
                Page.objects.filter(parent__isnull=True).order_by("-position").first()
            )
            self.position = (last_root.position + 1) if last_root else 0

        super().save(*args, **kwargs)

        # Only generate parts if not explicitly skipped
        if not self._skip_part_generation:
            self.generate_parts()

        # Reset the flag after save
        self._skip_part_generation = False

    def _process_behaviour_block(self, original_text, global_start):
        """
        Process a behaviour block by trimming blank lines and adjusting line numbers.

        Args:
            original_text: the exact substring from content (may include blank lines)
            global_start: the line number in the full document where original_text begins

        Returns:
            tuple: (adjusted_start, adjusted_end, stripped_content) or None if block is empty
        """
        lines = original_text.splitlines()
        first_idx = None
        last_idx = 0
        for i, line in enumerate(lines):
            if line.strip():
                if first_idx is None:
                    first_idx = i
                last_idx = i
        if first_idx is None:
            return None
        adjusted_start = global_start + first_idx
        adjusted_end = global_start + last_idx
        stripped_content = "\n".join(lines[first_idx : last_idx + 1])
        return adjusted_start, adjusted_end, stripped_content

    def _extract_code_blocks(self, pattern, content_type):
        """
        Extract code blocks (scenario/example) from content.

        Args:
            pattern: regex pattern to match the blocks
            content_type: ContentType enum value

        Returns:
            list: List of dictionaries containing block information
        """
        blocks = []
        for match in re.finditer(pattern, self.content, re.DOTALL):
            start_pos = match.start()
            start_line = self.content[:start_pos].count("\n")
            block_text = match.group(0)
            end_line = start_line + block_text.count("\n")
            block_content = match.group(1)
            blocks.append(
                {
                    "content": block_content,
                    "start_line": start_line,
                    "end_line": end_line,
                    "content_hash": Part._generate_hash(block_content),
                    "content_type": content_type,
                }
            )
        return blocks

    def _process_behaviour_blocks(self, all_block_matches):
        """
        Process behaviour blocks between code blocks.

        Args:
            all_block_matches: List of all matched code blocks

        Returns:
            list: List of dictionaries containing behaviour block information
        """
        behaviour_blocks = []

        # Process content before first block
        if all_block_matches:
            start_index = 0
            end_index = all_block_matches[0].start()
            original_text = self.content[start_index:end_index]
            processed = self._process_behaviour_block(original_text, 0)
            if processed:
                adj_start, adj_end, stripped_content = processed
                behaviour_blocks.append(
                    {
                        "content": stripped_content,
                        "start_line": adj_start,
                        "end_line": adj_end,
                        "content_hash": Part._generate_hash(stripped_content),
                        "content_type": ContentType.BEHAVIOUR,
                    }
                )
        else:
            # If no blocks, entire content is behaviour
            processed = self._process_behaviour_block(self.content, 0)
            if processed:
                adj_start, adj_end, stripped_content = processed
                behaviour_blocks.append(
                    {
                        "content": stripped_content,
                        "start_line": adj_start,
                        "end_line": adj_end,
                        "content_hash": Part._generate_hash(stripped_content),
                        "content_type": ContentType.BEHAVIOUR,
                    }
                )

        # Process content between blocks
        for i in range(len(all_block_matches) - 1):
            start_index = all_block_matches[i].end()
            end_index = all_block_matches[i + 1].start()
            original_text = self.content[start_index:end_index]
            global_start = self.content[:start_index].count("\n")
            processed = self._process_behaviour_block(original_text, global_start)
            if processed:
                adj_start, adj_end, stripped_content = processed
                behaviour_blocks.append(
                    {
                        "content": stripped_content,
                        "start_line": adj_start,
                        "end_line": adj_end,
                        "content_hash": Part._generate_hash(stripped_content),
                        "content_type": ContentType.BEHAVIOUR,
                    }
                )

        # Process content after last block
        if all_block_matches:
            start_index = all_block_matches[-1].end()
            end_index = len(self.content)
            original_text = self.content[start_index:end_index]
            global_start = self.content[:start_index].count("\n")
            processed = self._process_behaviour_block(original_text, global_start)
            if processed:
                adj_start, adj_end, stripped_content = processed
                behaviour_blocks.append(
                    {
                        "content": stripped_content,
                        "start_line": adj_start,
                        "end_line": adj_end,
                        "content_hash": Part._generate_hash(stripped_content),
                        "content_type": ContentType.BEHAVIOUR,
                    }
                )

        return behaviour_blocks

    def _update_parts(self, all_blocks):
        """
        Update or create parts based on the extracted blocks.

        Args:
            all_blocks: List of all blocks (behaviour, scenario, example)
        """
        new_part_hashes = {block["content_hash"] for block in all_blocks}

        # Delete existing parts not in new set
        if self.pk:
            self.parts.exclude(content_hash__in=new_part_hashes).delete()

        # Create or update parts
        for block in all_blocks:
            if block["end_line"] < block["start_line"]:
                logger.warning(
                    f"Invalid line numbers detected: start_line={block['start_line']}, "
                    f"end_line={block['end_line']} - adjusting"
                )
                block["end_line"] = block["start_line"] + 1

            self.parts.update_or_create(
                content_hash=block["content_hash"],
                defaults={
                    "start_line": block["start_line"],
                    "end_line": block["end_line"],
                    "content_type": block["content_type"],
                },
            )

    def generate_parts(self):
        """
        Generates parts from page content based on content type:
        - Behaviour: Text blocks between scenario and example blocks
        - Scenario: Content inside ```scenario blocks
        - Example: Content inside ```example blocks
        """
        if self._skip_part_generation:
            return

        try:
            # Define patterns for code blocks
            scenario_pattern = r"```\s*scena(?:r)?io\s*\n(.*?)\n\s*```"
            example_pattern = r"```\s*example\s*\n(.*?)\n\s*```"

            # Extract code blocks
            scenario_blocks = self._extract_code_blocks(
                scenario_pattern, ContentType.SCENARIO
            )
            example_blocks = self._extract_code_blocks(
                example_pattern, ContentType.EXAMPLE
            )

            # Get all blocks sorted by position
            all_block_matches = sorted(
                re.finditer(
                    f"({scenario_pattern})|({example_pattern})", self.content, re.DOTALL
                ),
                key=lambda m: m.start(),
            )

            # Process behaviour blocks
            behaviour_blocks = self._process_behaviour_blocks(all_block_matches)

            # Combine all blocks
            all_blocks = behaviour_blocks + scenario_blocks + example_blocks
            logger.info(
                f"Total blocks: {len(all_blocks)} (behaviour: {len(behaviour_blocks)}, "
                f"scenario: {len(scenario_blocks)}, example: {len(example_blocks)})"
            )

            # Update parts in database
            self._update_parts(all_blocks)

            logger.info(f"Part generation completed successfully for page {self.pk}")

        except Exception as e:
            logger.error(f"Part generation failed for page {self.pk}: {str(e)}")
            raise PartGenerationError(f"Failed to generate parts: {str(e)}") from e

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

    @classmethod
    def reorder_positions(cls, parent_id=None):
        """
        Reorders all pages at a given level (parent) to have consecutive positions.
        This ensures there are no gaps in positions after deletions or moves.

        Args:
            parent_id: The ID of the parent page, or None for root level pages
        """
        pages = cls.objects.filter(parent_id=parent_id).order_by("position")
        for index, page in enumerate(pages):
            if page.position != index:
                page.position = index
                page.save(update_fields=["position"])


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

    is_handover = models.BooleanField(
        default=False,
        help_text="Special tag for downstream processing (valid only for scenarios and examples)",
    )

    # Tracking
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)

    objects = PartManager()

    class Meta:
        ordering = ["start_line"]
        constraints = [
            models.UniqueConstraint(
                fields=["page", "start_line"],
                name="unique_part_per_page",
            )
        ]
        indexes = [
            models.Index(fields=["page", "content_type"]),
            models.Index(fields=["page", "content_hash"]),
        ]

    def clean(self):
        if self.is_handover and self.content_type == ContentType.BEHAVIOUR:
            raise ValidationError(
                {
                    "is_handover": "Handover tag can only be applied to scenarios and examples"
                }
            )
        if self.end_line < self.start_line:
            raise ValidationError(
                {"end_line": "End line must be greater than start line"}
            )

    def save(self, *args, **kwargs):
        self.full_clean()
        super().save(*args, **kwargs)

    @staticmethod
    def _generate_hash(content: str) -> str:
        """Generates a hash for the given content."""

        return sha256(content.encode()).hexdigest()

    def __str__(self):
        return f"{self.content_type}: {self.start_line} - {self.end_line} ({self.page.title})"


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

    # TODO: the PK should be the part hash
    # TODO: when retriving a vector, the auhtorization should be checked at page/part level.
    #  Multiple parts can have the same vector as long as they have the same part hash.
    # At the same time, multiple parts with the same hash may have different embedding settings.

    # TODO: to find a way to delete a vector when no parts with the same hash exist.
    part = models.ForeignKey(
        "Part", on_delete=models.SET_NULL, null=True, related_name="vectors"
    )

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
        # TODO: find a more robust solution once the TODOs above are addressed
        if self.part is None:
            return

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
