import uuid

from django.contrib.auth import get_user_model
from django.contrib.sites.models import Site
from django.core.mail import EmailMultiAlternatives
from django.db import models
from django.template.loader import render_to_string
from django.urls import reverse
from djstripe.models import Customer
from organizations.models import (
    AbstractOrganization,
    AbstractOrganizationUser,
    AbstractOrganizationOwner,
    AbstractOrganizationInvitation,
)


def generate_org_id():
    """Generate a custom ID for the organization."""
    return f"org-{uuid.uuid4()}"


class Organization(AbstractOrganization):
    """
    Custom organization model with a string primary key and Stripe integration.
    """

    id = models.CharField(
        primary_key=True,
        default=generate_org_id,
        max_length=40,
        editable=False,
    )
    stripe_customer = models.OneToOneField(
        Customer,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="hebo_organization",
    )

    class Meta(AbstractOrganization.Meta):
        abstract = False


class OrganizationUser(AbstractOrganizationUser):
    """
    Custom organization user model linking users to organizations.
    """

    organization = models.ForeignKey(
        Organization, on_delete=models.CASCADE, related_name="hebo_organization_user"
    )
    user = models.ForeignKey(
        "auth.User", on_delete=models.CASCADE, related_name="hebo_organization_user"
    )

    class Meta(AbstractOrganizationUser.Meta):
        abstract = False


class OrganizationOwner(AbstractOrganizationOwner):
    """
    Custom organization owner model designating an organization user as the owner.
    """

    organization_user = models.OneToOneField(
        OrganizationUser,
        on_delete=models.CASCADE,
        related_name="hebo_organization_owner",
    )
    organization = models.OneToOneField(
        Organization, on_delete=models.CASCADE, related_name="owner"
    )

    class Meta(AbstractOrganizationOwner.Meta):
        abstract = False


class OrganizationInvitation(AbstractOrganizationInvitation):
    """
    Custom organization invitation model with admin role support.
    """

    organization = models.ForeignKey(
        Organization,
        on_delete=models.CASCADE,
        related_name="hebo_organization_invitation",
    )
    # Use invitee_identifier from the AbstractOrganizationInvitation as the email field
    is_admin = models.BooleanField(default=False)

    class Meta(AbstractOrganizationInvitation.Meta):
        abstract = False

    @property
    def email(self):
        """Use invitee_identifier as the email field."""
        return str(self.invitee_identifier)

    @email.setter
    def email(self, value):
        """Set invitee_identifier when email is set."""
        self.invitee_identifier = value

    def send_invitation(self, request=None):
        """
        Send invitation email to the user.

        Extending the base behavior to provide custom email templates and handle
        the case where the user might not exist yet.
        """
        import logging

        logger = logging.getLogger(__name__)

        try:
            # Check if there's a site configured, if not use the request
            domain = None
            try:
                if Site._meta.installed:
                    site = Site.objects.get_current()
                    domain = site.domain
                    logger.debug(f"Using site domain: {domain}")
            except Exception as site_error:
                logger.warning(f"Error getting Site: {str(site_error)}")

            if not domain and request is not None:
                domain = request.get_host()
                logger.debug(f"Using request host: {domain}")
            elif not domain:
                domain = "localhost"  # Fallback domain
                logger.debug(f"Using fallback domain: {domain}")

            # Check if the user already exists
            User = get_user_model()
            try:
                user = User.objects.get(email=self.email)
                logger.debug(f"Found existing user for email: {self.email}")
                # Build the acceptance URL for existing users
                accept_url = "http://{}{}".format(
                    domain,
                    reverse(
                        "organization_invitation_accept",
                        kwargs={"invitation_id": self.id},
                    ),
                )
                register_url = None
            except User.DoesNotExist:
                user = None
                logger.debug(f"No existing user for email: {self.email}")
                # Build registration URL for new users
                accept_url = None
                register_url = (
                    "http://{}/accounts/signup/?invitation_id={}&email={}".format(
                        domain, str(self.id), str(self.email)
                    )
                )

            # Prepare context for email templates
            context = {
                "organization": self.organization,
                "invited_by": self.invited_by,
                "user": user,
                "email": str(self.email),
                "accept_url": accept_url,
                "register_url": register_url,
            }

            logger.debug(
                f"Email context prepared: accept_url={accept_url}, register_url={register_url}"
            )

            # Render both html and text templates
            subject = f"Invitation to join {self.organization.name}"
            text_content = render_to_string(
                "emails/organizations/invitation.txt", context
            )
            html_content = render_to_string(
                "emails/organizations/invitation.html", context
            )

            # Create email message
            msg = EmailMultiAlternatives(subject, text_content, to=[self.email])
            msg.attach_alternative(html_content, "text/html")
            msg.send()
            logger.info(f"Invitation email sent to {self.email}")

            return True
        except Exception as e:
            # Log the error, but don't crash
            logger.error(f"Error sending invitation: {str(e)}", exc_info=True)
            return False

    def accept(self, user):
        """
        Accept the invitation by creating an OrganizationUser for the user.
        """
        org_user = OrganizationUser.objects.create(
            organization=self.organization, user=user, is_admin=self.is_admin
        )

        # If this is the first user, make them the owner
        if not OrganizationOwner.objects.filter(
            organization=self.organization
        ).exists():
            OrganizationOwner.objects.create(
                organization=self.organization, organization_user=org_user
            )

        return org_user
