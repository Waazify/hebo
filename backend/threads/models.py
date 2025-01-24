import random
from typing import TYPE_CHECKING

from django.db import models, transaction
from django.utils.translation import gettext_lazy as _
from core.managers import OrganizationManagerMixin
from hebo_organizations.models import Organization
from django.core.exceptions import ValidationError
from django.core.serializers.json import DjangoJSONEncoder
from django.db.models.signals import pre_save
from django.dispatch import receiver

if TYPE_CHECKING:
    from django.db.models.manager import RelatedManager


class ThreadManager(OrganizationManagerMixin, models.Manager):
    pass


class Thread(models.Model):
    ADJECTIVES = [
        "Happy",
        "Clever",
        "Bright",
        "Swift",
        "Kind",
        "Wise",
        "Brave",
        "Calm",
        "Eager",
        "Fair",
        "Gentle",
        "Agile",
        "Bold",
        "Cheerful",
        "Daring",
        "Earnest",
        "Friendly",
        "Graceful",
        "Honest",
        "Inventive",
        "Jolly",
        "Keen",
        "Lively",
        "Merry",
        "Noble",
        "Optimistic",
        "Peaceful",
        "Quick",
        "Radiant",
        "Smart",
        "Thoughtful",
        "Upbeat",
        "Valiant",
        "Witty",
        "Zealous",
        "Adventurous",
        "Brilliant",
        "Caring",
        "Determined",
        "Energetic",
        "Faithful",
        "Generous",
        "Helpful",
        "Inspiring",
        "Joyful",
        "Knowledgeable",
        "Loving",
        "Mindful",
        "Nurturing",
        "Observant",
        "Patient",
        "Reliable",
        "Sincere",
        "Talented",
        "Understanding",
        "Vibrant",
        "Warm",
        "Youthful",
        "Ambitious",
        "Balanced",
        "Creative",
        "Dynamic",
        "Elegant",
        "Focused",
        "Genuine",
        "Harmonious",
        "Innovative",
    ]

    COLORS = [
        "Red",
        "Blue",
        "Green",
        "Yellow",
        "Purple",
        "Orange",
        "Pink",
        "Brown",
        "Gray",
        "Black",
        "White",
        "Teal",
        "Cyan",
        "Magenta",
        "Lime",
        "Indigo",
        "Violet",
        "Lavender",
        "Maroon",
        "Olive",
        "Navy",
        "Gold",
        "Silver",
        "Beige",
        "Turquoise",
        "Coral",
        "Salmon",
        "Plum",
        "Crimson",
        "Khaki",
        "Mustard",
        "Bronze",
        "Ivory",
        "Slate",
        "Charcoal",
        "Mint",
        "Emerald",
        "Sage",
        "Fuchsia",
        "Periwinkle",
        "Denim",
        "Taupe",
        "Cerulean",
        "Burgundy",
        "Russet",
        "Aquamarine",
    ]

    NOUNS = [
        "Panda",
        "Eagle",
        "Lion",
        "Dolphin",
        "Fox",
        "Owl",
        "Bear",
        "Wolf",
        "Tiger",
        "Hawk",
        "Elephant",
        "Giraffe",
        "Penguin",
        "Koala",
        "Kangaroo",
        "Cheetah",
        "Leopard",
        "Lynx",
        "Raccoon",
        "Deer",
        "Rabbit",
        "Squirrel",
        "Otter",
        "Seal",
        "Whale",
        "Gorilla",
        "Monkey",
        "Zebra",
        "Antelope",
        "Gazelle",
        "Hedgehog",
        "Badger",
        "Beaver",
        "Moose",
        "Bison",
        "Camel",
        "Llama",
        "Alpaca",
        "Puma",
        "Jaguar",
        "Rhinoceros",
        "Hippopotamus",
        "Crocodile",
        "Turtle",
        "Peacock",
        "Parrot",
        "Hummingbird",
        "Swan",
        "Falcon",
        "Raven",
        "Cardinal",
        "Flamingo",
        "Pelican",
        "Toucan",
        "Woodpecker",
        "Crane",
        "Dove",
        "Robin",
        "Sparrow",
        "Butterfly",
        "Dragonfly",
        "Ladybug",
        "Octopus",
        "Dolphin",
        "Seahorse",
        "Starfish",
        "Jellyfish",
        "Penguin",
    ]

    organization = models.ForeignKey(
        Organization,
        on_delete=models.CASCADE,
        related_name="threads",
        help_text=_("Organization this thread belongs to"),
    )
    is_open = models.BooleanField(
        default=True, help_text=_("Whether the thread is open or closed")
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    contact_name = models.CharField(
        max_length=100, help_text=_("Randomly generated name for the contact")
    )
    contact_identifier = models.CharField(max_length=100, null=True, blank=True)

    runs: "RelatedManager[Run]"  # Added for type hinting

    objects = ThreadManager()

    class Meta:
        ordering = ["-updated_at"]

    def save(self, *args, **kwargs):
        if not self.contact_name:
            self.contact_name = self._generate_contact_name()

        # Expire running runs when thread is updated
        if self.pk:  # Only for existing threads
            with transaction.atomic():
                self.runs.filter(status=Run.RunStatus.RUNNING).update(
                    status=Run.RunStatus.EXPIRED
                )
                super().save(*args, **kwargs)
        else:
            super().save(*args, **kwargs)

    def _generate_contact_name(self):
        """Generate a random contact name combining an adjective and a noun."""
        adjective = random.choice(self.ADJECTIVES)
        noun = random.choice(self.NOUNS)
        return f"{adjective} {noun}"


class MessageManager(OrganizationManagerMixin, models.Manager):
    def get_queryset(self):
        return super().get_queryset().select_related("thread__organization")


class Message(models.Model):
    class MessageType(models.TextChoices):
        AI = "ai", _("AI")
        HUMAN = "human", _("Human")
        HUMAN_AGENT = "human_agent", _("Human Agent")
        TOOL = "tool", _("Tool")
        COMMENT = "comment", _("Comment")

    thread = models.ForeignKey(
        Thread,
        on_delete=models.CASCADE,
        related_name="messages",
        help_text=_("The thread this message belongs to"),
    )
    created_at = models.DateTimeField(auto_now_add=True)
    message_type = models.CharField(
        max_length=20,
        choices=MessageType.choices,
        help_text=_("Type of the message sender"),
    )
    content = models.JSONField(
        encoder=DjangoJSONEncoder,
        help_text=_(
            "List of content objects. Example: [{'type': 'text', 'text': 'Hello'}]"
        ),
    )

    objects = MessageManager()

    class Meta:
        ordering = ["created_at"]
        indexes = [
            models.Index(fields=["thread", "created_at"]),
            models.Index(fields=["message_type"]),
        ]

    def clean(self):
        """Validate the content field structure."""
        if not isinstance(self.content, list):
            raise ValidationError({"content": _("Content must be a list of objects")})

        for item in self.content:
            if not isinstance(item, dict):
                raise ValidationError(
                    {"content": _("Each content item must be an object")}
                )
            if "type" not in item:
                raise ValidationError(
                    {"content": _("Each content item must have a 'type' field")}
                )

    def save(self, *args, **kwargs):
        self.clean()
        super().save(*args, **kwargs)

        # Update the thread's updated_at timestamp
        self.thread.save(update_fields=["updated_at"])


class SummaryManager(OrganizationManagerMixin, models.Manager):
    def get_queryset(self):
        return super().get_queryset().select_related("thread__organization")


class Summary(models.Model):
    """
    Stores a summary of a thread's conversation.
    """

    thread = models.OneToOneField(
        Thread,
        on_delete=models.CASCADE,
        related_name="summary",
        help_text=_("The thread this summary belongs to"),
    )
    content = models.TextField(
        help_text=_("The summarized content of the thread conversation")
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    objects = SummaryManager()

    class Meta:
        verbose_name_plural = "summaries"
        indexes = [
            models.Index(fields=["thread"]),
            models.Index(fields=["-updated_at"]),
        ]


class RunManager(OrganizationManagerMixin, models.Manager):
    def get_queryset(self):
        return super().get_queryset().select_related("thread__organization")


class Run(models.Model):
    class RunStatus(models.TextChoices):
        CREATED = "created", _("Created")
        RUNNING = "running", _("Running")
        COMPLETED = "completed", _("Completed")
        ERROR = "error", _("Error")
        EXPIRED = "expired", _("Expired")

    version = models.ForeignKey(
        "versions.Version",
        on_delete=models.CASCADE,
        related_name="runs",
        help_text=_("The version this run belongs to"),
    )

    thread = models.ForeignKey(
        Thread,
        on_delete=models.CASCADE,
        related_name="runs",
        help_text=_("The thread this run belongs to"),
    )
    status = models.CharField(
        max_length=20,
        choices=RunStatus.choices,
        default=RunStatus.CREATED,
        help_text=_("Current status of the run"),
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    objects = RunManager()


@receiver(pre_save, sender=Run)
def update_thread_on_run_creation(sender, instance, **kwargs):
    """
    Signal handler to update the thread's updated_at timestamp when a new run is created
    """
    if instance.pk is None:  # New run being created
        with transaction.atomic():
            # Update thread timestamp
            instance.thread.save(update_fields=["updated_at"])
