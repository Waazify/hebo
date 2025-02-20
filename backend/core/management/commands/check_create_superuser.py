from django.core.management.base import BaseCommand
from django.contrib.auth import get_user_model
import os


class Command(BaseCommand):
    help = "Creates a superuser if none exists"

    def handle(self, *args, **options):
        user = get_user_model()
        if not user.objects.filter(is_superuser=True).exists():
            if (
                (username := os.getenv("DJANGO_SUPERUSER_USERNAME"))
                and (email := os.getenv("DJANGO_SUPERUSER_EMAIL"))
                and (password := os.getenv("DJANGO_SUPERUSER_PASSWORD"))
            ):
                user.objects.create_superuser(
                    username=username, email=email, password=password
                )
                self.stdout.write(self.style.SUCCESS("Superuser created successfully"))
            else:
                self.stdout.write(self.style.WARNING("Superuser credentials not set"))
        else:
            self.stdout.write(self.style.SUCCESS("Superuser already exists"))
