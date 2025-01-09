from django.urls import path
from . import views

urlpatterns = [
    path("", views.AgentSettingUpdateView.as_view(), name="agent_setting_update"),
    path("tools/", views.ToolListView.as_view(), name="tool_list"),
    path("tools/create/", views.ToolCreateView.as_view(), name="tool_create"),
    path("tools/<int:pk>/", views.ToolUpdateView.as_view(), name="tool_update"),
    path("tools/<int:pk>/delete/", views.ToolDeleteView.as_view(), name="tool_delete"),
]
