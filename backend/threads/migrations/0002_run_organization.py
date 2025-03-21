# Generated by Django 5.1.4 on 2025-02-11 08:20

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("hebo_organizations", "0001_initial"),
        ("threads", "0001_initial"),
    ]

    operations = [
        migrations.AddField(
            model_name="run",
            name="organization",
            field=models.ForeignKey(
                default="org-5f7143a1-333a-4a54-985e-e85f7a2f9bad",
                help_text="Organization this run belongs to",
                on_delete=django.db.models.deletion.CASCADE,
                related_name="runs",
                to="hebo_organizations.organization",
            ),
            preserve_default=False,
        ),
    ]
