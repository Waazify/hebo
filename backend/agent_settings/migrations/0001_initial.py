# Generated by Django 5.1.4 on 2025-02-10 06:54

from django.db import migrations, models


class Migration(migrations.Migration):
    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="AgentSetting",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("delay", models.BooleanField(default=False)),
                ("hide_tool_messages", models.BooleanField(default=False)),
            ],
        ),
        migrations.CreateModel(
            name="LLMAdapter",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                (
                    "is_default",
                    models.BooleanField(
                        default=False,
                        help_text="If True, this adapter is available to all organizations",
                    ),
                ),
                (
                    "model_type",
                    models.CharField(
                        choices=[("chat", "Chat"), ("embedding", "Embedding")],
                        default="chat",
                        help_text="Type of model (chat, embedding, or vision)",
                        max_length=20,
                    ),
                ),
                (
                    "provider",
                    models.CharField(
                        choices=[
                            ("anthropic", "Anthropic"),
                            ("openai", "OpenAI"),
                            ("azure", "Azure"),
                            ("bedrock", "AWS Bedrock"),
                            ("vertex-ai", "Vertex AI"),
                            ("voyage", "Voyage"),
                        ],
                        help_text="AI provider for this model",
                        max_length=20,
                    ),
                ),
                (
                    "api_base",
                    models.URLField(
                        blank=True,
                        help_text="Base URL for API calls (if different from provider default)",
                    ),
                ),
                (
                    "name",
                    models.CharField(
                        help_text="Name/ID of the model from the provider",
                        max_length=100,
                    ),
                ),
                (
                    "aws_region",
                    models.CharField(
                        blank=True,
                        help_text="AWS region (required for Bedrock)",
                        max_length=50,
                    ),
                ),
                (
                    "api_key",
                    models.CharField(
                        blank=True,
                        help_text="API key, service account JSON, or other authentication credentials",
                        max_length=2000,
                    ),
                ),
                (
                    "aws_access_key_id",
                    models.CharField(
                        blank=True,
                        help_text="AWS Access Key ID (required for Bedrock)",
                        max_length=200,
                    ),
                ),
                (
                    "aws_secret_access_key",
                    models.CharField(
                        blank=True,
                        help_text="AWS Secret Access Key (required for Bedrock)",
                        max_length=200,
                    ),
                ),
            ],
        ),
        migrations.CreateModel(
            name="Tool",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("name", models.CharField(max_length=200)),
                (
                    "description",
                    models.TextField(help_text="Description of what the tool does"),
                ),
                (
                    "output_template",
                    models.TextField(help_text="Template to format the tool's output"),
                ),
                (
                    "tool_type",
                    models.CharField(
                        choices=[("action", "Action"), ("data_source", "Data Source")],
                        default="action",
                        max_length=20,
                    ),
                ),
                (
                    "openapi_url",
                    models.URLField(
                        blank=True,
                        help_text="OpenAPI URL for Action type tools",
                        null=True,
                    ),
                ),
                (
                    "auth_token",
                    models.CharField(
                        blank=True,
                        help_text="Optional authentication token for API",
                        max_length=255,
                        null=True,
                    ),
                ),
                (
                    "db_connection_string",
                    models.CharField(
                        blank=True,
                        help_text="Database connection string for Data Source type tools",
                        max_length=255,
                        null=True,
                    ),
                ),
                (
                    "query",
                    models.TextField(
                        blank=True,
                        help_text="Query to execute for Data Source type tools",
                        null=True,
                    ),
                ),
            ],
        ),
    ]
