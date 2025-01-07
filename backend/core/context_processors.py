from hebo_organizations.models import OrganizationUser, Organization
from knowledge.models import Page
from versions.models import Agent, Version


def organization_context(request):
    context = {
        "user_organization": None | Organization,
        "selected_agent": None | Agent,
        "selected_version": None | Version,
    }

    if request.user.is_authenticated:
        org_user = OrganizationUser.objects.filter(user=request.user).first()
        if org_user:
            context["user_organization"] = org_user.organization

            # Get agent_id and version_id from session
            agent_id = request.session.get("selected_agent_id")
            version_id = request.session.get("selected_version_id")

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
                    # Try to get specific version from session first
                    version = None
                    if version_id:
                        version = Version.objects.filter(agent=agent, id=version_id).first()
                    
                    # Fall back to current version or most recent
                    if not version:
                        version = (
                            Version.objects.filter(agent=agent, status="CURRENT").first()
                            or Version.objects.filter(agent=agent)
                            .order_by("-created_at")
                            .first()
                        )
                    
                    context["selected_version"] = version
                    # Store in session for future requests
                    request.session["selected_agent_id"] = agent.pk
                    if version:
                        request.session["selected_version_id"] = version.pk

            except (Agent.DoesNotExist, ValueError):
                pass

    return context


def knowledge_context(request):
    context = {
        "pages": Page.objects.filter(version__id=request.session.get("selected_version_id")).order_by("created_at"),
    }
    return context
