from typing import TypedDict, Optional

from django.conf import settings

from api_keys.models import APIKey
from hebo_organizations.models import OrganizationUser, Organization
from knowledge.models import Page
from versions.models import Agent, Version, VersionSlug


class OrganizationContext(TypedDict):
    user_organization: Optional[Organization]
    selected_agent: Optional[Agent]
    selected_version: Optional[Version]
    selected_version_slug: Optional[str]


def organization_context(request):
    context: OrganizationContext = {
        "user_organization": None,
        "selected_agent": None,
        "selected_version": None,
        "selected_version_slug": None,
    }

    if request.user.is_authenticated:
        org_user = OrganizationUser.objects.filter(user=request.user).first()
        if org_user:
            context["user_organization"] = org_user.organization

            # Get agent_id and version_id from session
            agent_id = request.session.get("selected_agent_id")
            version_id = request.session.get("selected_version_id")
            version_slug_id = request.session.get("selected_version_slug_id")

            try:
                agent = None
                if agent_id:
                    agent = Agent.objects.filter(
                        organization=org_user.organization, id=agent_id
                    ).first()

                if not agent:
                    agent = (
                        Agent.objects.filter(organization=org_user.organization)
                        .order_by("-created_at")
                        .first()
                    )

                if agent:
                    context["selected_agent"] = agent
                    request.session["selected_agent_id"] = agent.pk

                    # Try to get specific version from session first
                    version = None
                    if version_id:
                        version = Version.objects.filter(
                            agent=agent, id=version_id
                        ).first()

                    # Fall back to current version or most recent
                    if not version:
                        version = (
                            Version.objects.filter(
                                agent=agent, status="CURRENT"
                            ).first()
                            or Version.objects.filter(agent=agent)
                            .order_by("-created_at")
                            .first()
                        )

                    if version:
                        context["selected_version"] = version
                        request.session["selected_version_id"] = version.pk

                        version_slug = VersionSlug.objects.filter(
                            version=version
                        ).first()

                        if version_slug_id:
                            version_slug = VersionSlug.objects.filter(
                                id=version_slug_id
                            ).first()

                        if not version_slug:
                            version_slug = VersionSlug.objects.filter(
                                version=version
                            ).first()

                        if version_slug:
                            context["selected_version_slug"] = version_slug.slug
                            request.session["selected_version_slug_id"] = (
                                version_slug.pk
                            )

            except (Agent.DoesNotExist, ValueError):
                pass

    return context


def knowledge_context(request):
    context = {
        "pages": Page.objects.filter(
            version__id=request.session.get("selected_version_id")
        ).order_by("created_at"),
    }
    return context


def proxy_context(request):
    if not request.user.is_authenticated:
        return {
            "PROXY_SERVER_BASE_URL": settings.PROXY_SERVER_BASE_URL,
            "PROXY_SERVER_API_KEY": None,
        }

    """Add proxy server URL to template context."""
    org_user = OrganizationUser.objects.filter(user=request.user).first()
    if org_user:
        api_key = APIKey.objects.filter(organization=org_user.organization).first()
        return {
            "PROXY_SERVER_BASE_URL": settings.PROXY_SERVER_BASE_URL,
            "PROXY_SERVER_API_KEY": api_key.key if api_key else None,
        }
    return {
        "PROXY_SERVER_BASE_URL": settings.PROXY_SERVER_BASE_URL,
        "PROXY_SERVER_API_KEY": None,
    }
