from django.apps import AppConfig


class HeboOrganizationsConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "hebo_organizations"

    def ready(self):
        try:
            import hebo_organizations.signals  # noqa
        except ImportError:
            pass
