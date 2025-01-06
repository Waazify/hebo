from django.urls import path
from . import views

urlpatterns = [
    path("", views.AgentListView.as_view(), name="agent_list"),
    path("create/", views.AgentInlineCreateView.as_view(), name="agent_inline_create"),
    path("<int:pk>/inline-edit/", views.AgentInlineUpdateView.as_view(), name="agent_inline_update"),
    path(
        "<int:agent_pk>/versions/",
        views.VersionListView.as_view(),
        name="version_list",
    ),
    path(
        "versions/<int:pk>/delete/",
        views.VersionDeleteView.as_view(),
        name="version_delete",
    ),
    path(
        '<int:agent_pk>/versions/<int:version_pk>/set-active/',
        views.SetActiveVersionView.as_view(),
        name='set_active_version'
    ),
]
