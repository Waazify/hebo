from django.db import models
from django.utils import timezone
from django.utils.translation import gettext_lazy as _
from versions.models import Version
from hebo_organizations.models import Organization
from core.managers import OrganizationManagerMixin


class TestCaseManager(OrganizationManagerMixin, models.Manager):
    pass


class TestCase(models.Model):
    """
    Represents a single test case.
    """

    TEST_STATUS = [
        ("passed", "Passed"),
        ("failed", "Failed"),
        ("skipped", "Skipped"),
    ]

    organization = models.ForeignKey(
        Organization,
        on_delete=models.CASCADE,
        related_name="test_cases",
        help_text=_("Organization this test case belongs to"),
    )
    version = models.ForeignKey(
        Version,
        related_name="test_cases",
        on_delete=models.CASCADE,
        help_text=_("The version this test case belongs to"),
    )

    name = models.CharField(max_length=100)
    description = models.TextField(null=True, blank=True)
    status = models.CharField(max_length=10, choices=TEST_STATUS, default="skipped")

    # Metadata fields
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)

    objects = TestCaseManager()

    class Meta:
        ordering = ["-updated_at"]
        verbose_name = "Test Case"
        verbose_name_plural = "Test Cases"

    def __str__(self):
        return self.name


class TestRunManager(OrganizationManagerMixin, models.Manager):
    def get_queryset(self):
        return super().get_queryset().select_related("version__organization")


class TestRun(models.Model):
    """
    Represents metrics and results for a test run of test cases.
    """

    organization = models.ForeignKey(
        Organization,
        on_delete=models.CASCADE,
        related_name="test_runs",
        help_text=_("Organization this test run belongs to"),
    )
    version = models.OneToOneField(
        Version,
        related_name="test_run",
        on_delete=models.CASCADE,
        help_text=_("The version this test run belongs to"),
    )

    # Metrics fields
    context_precision = models.FloatField()
    context_recall = models.FloatField()
    tool_call_accuracy = models.FloatField()
    factual_correctness = models.FloatField()
    faithfulness = models.FloatField()
    response_relevance = models.FloatField()
    evaluation_plan_strength = models.CharField(max_length=50)

    # Metadata fields
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)

    objects = TestRunManager()

    class Meta:
        ordering = ["-updated_at"]
        verbose_name = "Test Run"
        verbose_name_plural = "Test Runs"

    def __str__(self):
        return f"Test Run for Version {self.version.name}"

    @property
    def total_test_cases(self):
        return TestRunCase.objects.filter(test_run=self).count()

    @property
    def passed_test_cases(self):
        return TestRunCase.objects.filter(test_run=self, status="passed").count()

    @property
    def failed_test_cases(self):
        return TestRunCase.objects.filter(test_run=self, status="failed").count()

    @property
    def skipped_test_cases(self):
        return TestRunCase.objects.filter(test_run=self, status="skipped").count()


class HintManager(OrganizationManagerMixin, models.Manager):
    def get_queryset(self):
        return super().get_queryset().select_related("test_run__version__organization")


class Hint(models.Model):
    """
    Represents a hint to improve the test run.
    """

    test_run = models.ForeignKey(
        TestRun,
        related_name="hints",
        on_delete=models.CASCADE,
        help_text=_("The test run this hint belongs to"),
    )
    content = models.TextField()

    # Metadata fields
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)

    objects = HintManager()

    class Meta:
        ordering = ["-updated_at"]
        verbose_name = "Hint"
        verbose_name_plural = "Hints"

    def __str__(self):
        return f"Hint for {self.test_run.version.name}"


class TestRunCase(models.Model):
    """
    Represents the relationship between a TestRun and a TestCase,
    storing run-specific test case results.
    """

    test_run = models.ForeignKey(
        TestRun, related_name="test_run_cases", on_delete=models.CASCADE
    )
    test_case = models.ForeignKey(
        TestCase, related_name="test_run_cases", on_delete=models.CASCADE
    )
    status = models.CharField(
        max_length=10, choices=TestCase.TEST_STATUS, default="skipped"
    )
    executed_at = models.DateTimeField(auto_now_add=True)
    execution_time = models.FloatField(null=True, blank=True)
    error_message = models.TextField(null=True, blank=True)

    class Meta:
        unique_together = ["test_run", "test_case"]
