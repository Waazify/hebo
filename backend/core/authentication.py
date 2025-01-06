from rest_framework.authentication import BaseAuthentication
from rest_framework.exceptions import AuthenticationFailed
from api_keys.models import APIKey
from django.utils import timezone


class APIKeyAuthentication(BaseAuthentication):
    def authenticate(self, request):
        api_key = request.META.get("HTTP_X_API_KEY") or request.query_params.get(
            "api_key"
        )

        if not api_key:
            return None

        try:
            key = APIKey.objects.select_related("organization").get(
                key=api_key, is_active=True
            )
            # Update last used timestamp
            key.last_used_at = timezone.now()
            key.save(update_fields=["last_used_at"])

            # Return tuple of (user, auth)
            # In this case, we're using the organization as the auth object
            return (None, key.organization)
        except APIKey.DoesNotExist:
            raise AuthenticationFailed("Invalid API key")

    def authenticate_header(self, request):
        return "X-API-Key"
