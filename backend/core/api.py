from rest_framework.routers import DefaultRouter
from knowledge import views as knowledge_views
from agent_settings import views as agent_settings_views

router = DefaultRouter()

router.register(r"knowledge", knowledge_views.PageViewSet, basename="api-knowledge")
router.register(r"tools", agent_settings_views.ToolViewSet, basename="api-tools")
router.register(
    r"agent-settings",
    agent_settings_views.AgentSettingViewSet,
    basename="api-agent-settings",
)
