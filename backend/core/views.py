import logging

from allauth.account.views import SignupView
from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.db import connections
from django.db.utils import OperationalError
from django.http import JsonResponse
from django.shortcuts import redirect
from django.urls import reverse_lazy
from django.views.decorators.http import require_GET
from hebo_organizations.models import OrganizationUser

logger = logging.getLogger(__name__)


@login_required
@require_GET
def home(request):
    if not request.user.hebo_organization_user.exists():
        return redirect("organization_create")

    user_organization = (
        OrganizationUser.objects.filter(user=request.user).first().organization
    )

    return redirect("knowledge_list", organization_pk=user_organization.pk)


def health_check(request):
    """
    Health check endpoint that verifies database connectivity
    and returns application status.
    """
    # Check database connectivity
    db_status = "healthy"
    try:
        connections["default"].ensure_connection()
    except OperationalError:
        db_status = "unhealthy"

    # Build response data
    health_data = {
        "status": "healthy" if db_status == "healthy" else "unhealthy",
        "version": settings.APP_VERSION,
        "database": db_status,
    }

    status_code = 200 if db_status == "healthy" else 503
    return JsonResponse(health_data, status=status_code)


class CustomSignupView(SignupView):
    """
    Custom signup view that captures invitation parameters from the URL.
    """

    def get(self, request, *args, **kwargs):
        # Check for invitation parameters
        invitation_id = request.GET.get("invitation_id")
        email = request.GET.get("email")

        logger.debug(
            f"SignupView GET with invitation_id={invitation_id}, email={email}"
        )

        if invitation_id:
            # Store the invitation ID in the session as a string
            # Make sure it's a string to avoid serialization issues
            request.session["invitation_id"] = str(invitation_id)

            # Pre-fill the email field if provided
            if email:
                self.initial["email"] = str(email)

        return super().get(request, *args, **kwargs)

    def post(self, request, *args, **kwargs):
        # Ensure all session data is serializable before processing the form
        if "invitation_id" in request.session:
            invitation_id = request.session["invitation_id"]
            request.session["invitation_id"] = str(invitation_id)
            logger.debug(f"Ensuring invitation_id is serializable: {invitation_id}")

        # Continue with normal form processing
        return super().post(request, *args, **kwargs)

    def get_success_url(self):
        # If there was an invitation, redirect to the organization settings
        invitation_id = self.request.session.get("invitation_id")
        if invitation_id:
            from hebo_organizations.models import OrganizationInvitation

            try:
                invitation = OrganizationInvitation.objects.get(pk=invitation_id)
                return reverse_lazy(
                    "organization_settings",
                    kwargs={"organization_pk": invitation.organization.pk},
                )
            except OrganizationInvitation.DoesNotExist:
                logger.warning(f"Invitation {invitation_id} not found")
                pass

        # Otherwise use the default success URL
        return super().get_success_url()
