"""
Custom adapter for Django Allauth to ensure
all session data is properly serializable.
"""

from allauth.account.adapter import DefaultAccountAdapter
import logging

logger = logging.getLogger(__name__)


class CustomAccountAdapter(DefaultAccountAdapter):
    """
    Custom account adapter that ensures session data is JSON serializable.
    """

    def __init__(self, request=None):
        super().__init__(request)
        self.clean_session_data()

    def clean_session_data(self):
        """
        Ensure invitation-related session data is serializable.
        Skip all social auth related keys for safety.
        """
        if not self.request or not hasattr(self.request, "session"):
            return

        # Keys that we know might need cleaning during the invitation flow
        invitation_keys = ["invitation_id", "email"]

        try:
            if hasattr(self.request.session, "keys") and callable(
                self.request.session.keys
            ):
                for key in list(self.request.session.keys()):
                    # Skip all social auth keys
                    if key.startswith("socialaccount_"):
                        continue

                    # Focus on invitation-related keys and any obvious proxy objects
                    try:
                        value = self.request.session[key]
                        if key in invitation_keys or (
                            hasattr(value, "__class__")
                            and value.__class__.__name__ == "__proxy__"
                        ):
                            self.request.session[key] = str(value)
                            logger.debug(f"Cleaned session key {key}")
                    except Exception as e:
                        logger.warning(f"Error handling session key {key}: {str(e)}")
        except Exception as e:
            logger.error(f"Error cleaning session data: {str(e)}")

    def pre_save(self, request, user, *args, **kwargs):
        """Clean session before saving user."""
        self.clean_session_data()
        return super().pre_save(request, user, *args, **kwargs)  # type: ignore

    def save_user(self, request, user, form, commit=True):
        """Clean session before and after saving user."""
        self.clean_session_data()
        user = super().save_user(request, user, form, commit)
        self.clean_session_data()
        return user

    def login(self, request, user):
        """Clean session before logging in user."""
        self.clean_session_data()
        return super().login(request, user)
