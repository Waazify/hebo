from django.urls import path
from . import views

urlpatterns = [
    path(
        "create/",
        views.OrganizationCreateView.as_view(),
        name="organization_create",
    ),
    path(
        "<str:organization_pk>/delete/",
        views.OrganizationDeleteView.as_view(),
        name="organization_delete",
    ),
    path(
        "<str:organization_pk>/settings/",
        views.OrganizationSettingsView.as_view(),
        name="organization_settings",
    ),
    path(
        "settings/",
        views.OrganizationSettingsView.as_view(),
        name="organization_settings_base",
    ),
]
