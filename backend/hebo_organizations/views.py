import json
import logging

from django.contrib.auth.mixins import LoginRequiredMixin
from django.urls import reverse_lazy
from django.views.generic import UpdateView, DeleteView, CreateView
from django.shortcuts import redirect, get_object_or_404
from django.db import transaction
from django.core.exceptions import PermissionDenied
from django.http import JsonResponse
from django.contrib.auth import get_user_model
from django.views import View
from django.contrib import messages
from django.urls import reverse

from .models import (
    Organization,
    OrganizationUser,
    OrganizationOwner,
    OrganizationInvitation,
)
from core.mixins import OrganizationPermissionMixin


logger = logging.getLogger(__name__)


class OrganizationSettingsView(
    LoginRequiredMixin, OrganizationPermissionMixin, UpdateView
):
    model = Organization
    template_name = "organizations/settings.html"
    fields = ["name"]
    context_object_name = "organization"
    require_organization = (
        False  # Allow access without organization for base settings URL
    )

    def get_object(self, queryset=None):
        return self.organization if self.organization else None

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context["has_organization"] = Organization.objects.filter(
            hebo_organization_user__user=self.request.user
        ).exists()
        # Add is_owner flag to context for template use
        if self.organization:
            context["is_owner"] = self.organization.is_owner(self.request.user)
        return context

    def post(self, request, *args, **kwargs):
        # Check if user is owner before allowing updates
        if self.organization and not self.organization.is_owner(request.user):
            raise PermissionDenied(
                "You must be an owner to update organization settings"
            )
        return super().post(request, *args, **kwargs)

    def get_success_url(self):
        return reverse_lazy(
            "organization_settings",
            kwargs={"organization_pk": self.object.pk},  # type: ignore
        )


class OrganizationCreateView(LoginRequiredMixin, CreateView):
    model = Organization
    fields = ["name"]
    template_name = "organizations/settings.html"

    def form_valid(self, form):
        with transaction.atomic():
            organization = form.save()

            org_user = OrganizationUser.objects.create(
                organization=organization, user=self.request.user
            )

            OrganizationOwner.objects.create(
                organization_user=org_user, organization=organization
            )

            self.object = organization

        return redirect(self.get_success_url())

    def get_success_url(self):
        return reverse_lazy(
            "organization_settings", kwargs={"organization_pk": self.object.pk}
        )


class OrganizationDeleteView(
    LoginRequiredMixin, OrganizationPermissionMixin, DeleteView
):
    model = Organization
    template_name = "organizations/confirm_delete.html"
    success_url = reverse_lazy("organization_settings_base")
    owner_required = True  # Only owners can delete organizations

    def get_object(self, queryset=None):
        return self.organization if self.organization else None


# Team Management API Views
class OrganizationMembersView(LoginRequiredMixin, OrganizationPermissionMixin, View):
    """API view to list organization members"""

    def get(self, request, *args, **kwargs):
        organization = self.organization
        members = []

        # Get all organization users
        org_users = OrganizationUser.objects.filter(
            organization=organization
        ).select_related("user")

        # Check if there's an owner
        try:
            owner = OrganizationOwner.objects.get(organization=organization)
            owner_user_id = owner.organization_user.user_id
        except OrganizationOwner.DoesNotExist:
            owner_user_id = None

        # Format member data
        for org_user in org_users:
            user = org_user.user
            members.append(
                {
                    "id": org_user.id,
                    "user_id": user.id,
                    "name": f"{user.first_name} {user.last_name}".strip()
                    or user.username,
                    "email": user.email,
                    "is_admin": org_user.is_admin,
                    "is_owner": user.id == owner_user_id,
                }
            )

        return JsonResponse({"members": members})


class OrganizationMemberUpdateView(
    LoginRequiredMixin, OrganizationPermissionMixin, View
):
    """API view to update a member's role"""

    owner_required = True  # Only owners can update member roles

    def patch(self, request, *args, **kwargs):
        organization = self.organization
        member_id = kwargs.get("member_id")

        try:
            # Parse request body
            data = json.loads(request.body)
            is_admin = data.get("is_admin", False)

            # Update the member
            org_user = OrganizationUser.objects.get(
                pk=member_id, organization=organization
            )
            org_user.is_admin = is_admin
            org_user.save()

            return JsonResponse({"success": True})
        except OrganizationUser.DoesNotExist:
            return JsonResponse({"error": "Member not found"}, status=404)
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=400)


class OrganizationMemberRemoveView(
    LoginRequiredMixin, OrganizationPermissionMixin, View
):
    """API view to remove a member from the organization"""

    def delete(self, request, *args, **kwargs):
        organization = self.organization
        member_id = kwargs.get("member_id")

        try:
            # Get the org user to remove
            org_user = OrganizationUser.objects.get(
                pk=member_id, organization=organization
            )

            # Check if user is owner - owners can't be removed
            try:
                owner = OrganizationOwner.objects.get(organization=organization)
                if owner.organization_user_id == org_user.id:
                    return JsonResponse(
                        {"error": "Cannot remove the organization owner"}, status=400
                    )
            except OrganizationOwner.DoesNotExist:
                pass

            # Check if user is admin and trying to remove another admin
            current_user_org = OrganizationUser.objects.get(
                organization=organization, user=request.user
            )

            if (
                not organization.is_owner(request.user)
                and current_user_org.is_admin
                and org_user.is_admin
            ):
                return JsonResponse(
                    {"error": "Admins cannot remove other admins"}, status=403
                )

            # Remove the user
            org_user.delete()
            return JsonResponse({"success": True})
        except OrganizationUser.DoesNotExist:
            return JsonResponse({"error": "Member not found"}, status=404)
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=400)


class OrganizationInvitationsView(
    LoginRequiredMixin, OrganizationPermissionMixin, View
):
    """API view to list pending invitations"""

    def get(self, request, *args, **kwargs):
        organization = self.organization

        # Get all pending invitations
        invitations = OrganizationInvitation.objects.filter(
            organization=organization
        ).order_by("-created")

        # Format invitation data
        invitations_data = [
            {
                "id": inv.id,
                "email": str(
                    inv.email
                ),  # Convert to string to ensure JSON serializable
                "created": inv.created.isoformat(),
            }
            for inv in invitations
        ]

        return JsonResponse({"invitations": invitations_data})


class OrganizationInvitationSendView(
    LoginRequiredMixin, OrganizationPermissionMixin, View
):
    """API view to send a new invitation"""

    def post(self, request, *args, **kwargs):
        organization = self.organization

        try:
            # Parse request body
            data = json.loads(request.body)
            email = str(data.get("email", ""))  # Ensure email is a string
            is_admin = bool(data.get("is_admin", False))  # Ensure is_admin is a boolean

            logger.debug(f"Invitation request: email={email}, is_admin={is_admin}")

            if not email:
                return JsonResponse({"error": "Email is required"}, status=400)

            # Check if user is already a member
            User = get_user_model()
            try:
                user = User.objects.get(email=email)
                if OrganizationUser.objects.filter(
                    organization=organization, user=user
                ).exists():
                    return JsonResponse(
                        {"error": "User is already a member"}, status=400
                    )
            except User.DoesNotExist:
                pass

            # Check if invitation already exists
            if OrganizationInvitation.objects.filter(
                organization=organization, invitee_identifier=email
            ).exists():
                return JsonResponse({"error": "Invitation already sent"}, status=400)

            # Create and send invitation
            invitation = OrganizationInvitation.objects.create(
                organization=organization,
                invitee_identifier=email,
                is_admin=is_admin,
                invited_by=request.user,
            )

            logger.debug(f"Created invitation: id={invitation.id}, email={email}")

            if not invitation.send_invitation(request):
                logger.error(f"Failed to send invitation email to {email}")
                return JsonResponse(
                    {"error": "Failed to send invitation email"}, status=500
                )

            return JsonResponse(
                {
                    "id": invitation.id,
                    "email": str(
                        invitation.email
                    ),  # Ensure email is serialized as a string
                    "created": invitation.created.isoformat(),
                }
            )

        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON"}, status=400)
        except Exception as e:
            logger.error(f"Error creating invitation: {str(e)}", exc_info=True)
            return JsonResponse({"error": str(e)}, status=500)


class OrganizationInvitationCancelView(
    LoginRequiredMixin, OrganizationPermissionMixin, View
):
    """API view to cancel a pending invitation"""

    def delete(self, request, *args, **kwargs):
        organization = self.organization
        invitation_id = kwargs.get("invitation_id")

        try:
            # Get the invitation
            invitation = OrganizationInvitation.objects.get(
                pk=invitation_id, organization=organization
            )

            # Delete the invitation
            invitation.delete()
            return JsonResponse({"success": True})
        except OrganizationInvitation.DoesNotExist:
            return JsonResponse({"error": "Invitation not found"}, status=404)
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=400)


# Invitation acceptance view
class OrganizationInvitationAcceptView(LoginRequiredMixin, View):
    """View to accept an organization invitation"""

    def get(self, request, *args, **kwargs):
        invitation_id = kwargs.get("invitation_id")

        try:
            # Get the invitation
            invitation = get_object_or_404(OrganizationInvitation, pk=invitation_id)

            # Check if the invitation email matches the user's email
            # Convert to string to ensure it's serializable
            if str(invitation.email).lower() != str(request.user.email).lower():
                messages.error(
                    request, "This invitation was sent to a different email address."
                )
                return redirect("home")

            # Accept the invitation - this creates the OrganizationUser
            invitation.accept(request.user)

            # Delete the invitation since it's been accepted
            invitation.delete()

            messages.success(
                request, f"You have joined {invitation.organization.name}!"
            )
            return redirect(
                reverse(
                    "organization_settings",
                    kwargs={"organization_pk": invitation.organization.pk},
                )
            )

        except OrganizationInvitation.DoesNotExist:
            messages.error(request, "The invitation is no longer valid.")
            return redirect("home")
        except Exception as e:
            messages.error(request, f"Error accepting invitation: {str(e)}")
            return redirect("home")
