from django.dispatch import receiver
from django.contrib.auth import get_user_model
from allauth.account.signals import user_signed_up
from .models import (
    Organization,
    OrganizationUser,
    OrganizationOwner,
    OrganizationInvitation,
)
import logging

logger = logging.getLogger(__name__)
User = get_user_model()


@receiver(user_signed_up)
def create_default_organization(sender, request, user, **kwargs):
    """
    Handle user signup by either accepting an invitation or creating a default organization.
    """
    # Check if there's an invitation_id in the request
    invitation_id = None
    if hasattr(request, "session") and "invitation_id" in request.session:
        invitation_id = request.session.get("invitation_id")
        # Ensure it's a string
        invitation_id = str(invitation_id) if invitation_id else None
        logger.debug(f"Found invitation_id in session: {invitation_id}")

    # Clean any non-string values from session
    try:
        if hasattr(request, "session"):
            for key in list(request.session.keys()):
                if not isinstance(
                    request.session[key],
                    (str, int, float, bool, list, dict, type(None)),
                ):
                    logger.warning(
                        f"Removing non-serializable value from session key: {key}"
                    )
                    del request.session[key]
    except Exception as e:
        logger.error(f"Error cleaning session: {str(e)}")

    if invitation_id:
        try:
            # Try to find the invitation
            invitation = OrganizationInvitation.objects.get(pk=invitation_id)
            logger.debug(f"Found invitation for email: {invitation.email}")

            # Ensure we're comparing string values
            invitation_email = str(invitation.email).lower()
            user_email = str(user.email).lower()

            # Check if the invitation email matches the user's email
            if invitation_email == user_email:
                logger.debug(f"Accepting invitation for user: {user_email}")

                # Accept the invitation
                invitation.accept(user)

                # Delete the invitation
                invitation.delete()

                # Clear the session
                if hasattr(request, "session") and "invitation_id" in request.session:
                    del request.session["invitation_id"]

                # Don't create a default organization since the user joined an existing one
                return
            else:
                logger.warning(
                    f"Email mismatch: invitation {invitation_email} vs user {user_email}"
                )
        except OrganizationInvitation.DoesNotExist:
            logger.warning(f"Invitation not found: {invitation_id}")
            # If invitation doesn't exist, continue with default organization creation
            pass
        except Exception as e:
            logger.error(f"Error processing invitation: {str(e)}")
            # On error, still create default organization

    # Only create organization if user doesn't have one
    if not OrganizationUser.objects.filter(user=user).exists():
        # Create organization name using first_name or email
        org_name = (
            f"{user.first_name}'s Organization"
            if user.first_name
            else f"{user.email.split('@')[0]}'s Organization"
        )

        logger.debug(
            f"Creating default organization: {org_name} for user: {user.email}"
        )

        # Create the organization
        org = Organization.objects.create(name=org_name)

        # Create organization user
        org_user = OrganizationUser.objects.create(
            organization=org, user=user, is_admin=True
        )

        # Create organization owner
        OrganizationOwner.objects.create(organization=org, organization_user=org_user)
