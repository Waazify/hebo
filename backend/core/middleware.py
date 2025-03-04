import logging
from django.utils.deprecation import MiddlewareMixin

logger = logging.getLogger(__name__)


class SessionCleaningMiddleware(MiddlewareMixin):
    """
    Middleware that ensures all session data is serializable before
    Django attempts to save it to the database, while preserving important
    structured data like OAuth2 state dictionaries.
    """

    def __init__(self, get_response=None):
        super().__init__(get_response)
        # Path prefixes that should be excluded from session cleaning
        self.excluded_paths = [
            "/accounts/google/login",
            "/accounts/github/login",
            # Add other social auth paths if needed
        ]

    def is_excluded_path(self, path):
        """Check if the current path should be excluded from session cleaning."""
        if not path:
            return False

        for excluded in self.excluded_paths:
            if path.startswith(excluded):
                return True
        return False

    def clean_value(self, value):
        """Clean a single value, converting proxy objects to strings."""
        if hasattr(value, "__class__") and value.__class__.__name__ == "__proxy__":
            return str(value)
        return value

    def clean_dict_recursively(self, data):
        """Clean dictionary values recursively."""
        result = {}
        for k, v in data.items():
            # Ensure key is a string
            key = str(k)

            if isinstance(v, dict):
                # Recursively clean nested dictionaries
                result[key] = self.clean_dict_recursively(v)
            elif isinstance(v, list):
                # Recursively clean lists
                result[key] = [
                    (
                        self.clean_dict_recursively(item)
                        if isinstance(item, dict)
                        else self.clean_value(item)
                    )
                    for item in v
                ]
            else:
                # Clean other values
                result[key] = self.clean_value(v)
        return result

    def process_response(self, request, response):
        """Clean session data before Django saves it, but skip for excluded paths."""
        if not hasattr(request, "session") or not request.session.modified:
            return response

        # Skip session cleaning completely for social auth paths
        if hasattr(request, "path") and self.is_excluded_path(request.path):
            logger.debug(f"Skipping session cleaning for excluded path: {request.path}")
            return response

        # For all other paths, clean the session to ensure it's serializable
        try:
            # Special handling for any keys we know might contain proxies
            for key in list(request.session.keys()):
                # Skip socialaccount keys entirely for safety
                if key.startswith("socialaccount_"):
                    continue

                value = request.session[key]
                if isinstance(value, dict):
                    request.session[key] = self.clean_dict_recursively(value)
                else:
                    request.session[key] = self.clean_value(value)
        except Exception as e:
            logger.error(f"Error cleaning session data: {str(e)}")

        return response
