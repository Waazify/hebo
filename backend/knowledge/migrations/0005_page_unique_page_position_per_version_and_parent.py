# Generated by Django 5.1.4 on 2025-03-21 03:44

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("hebo_organizations", "0002_organizationinvitation_is_admin"),
        ("knowledge", "0004_remove_vectorstore_vector_index_and_more"),
        ("versions", "0003_versionslug_unique_version_slug_per_organization"),
    ]

    operations = [
        migrations.AddConstraint(
            model_name="page",
            constraint=models.UniqueConstraint(
                fields=("version", "parent", "position"),
                name="unique_page_position_per_version_and_parent",
            ),
        ),
    ]
