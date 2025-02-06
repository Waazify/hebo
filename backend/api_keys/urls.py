from django.urls import path
from . import views

urlpatterns = [
    path("", views.APIKeyListView.as_view(), name="api_key_list"),
    path("create/", views.APIKeyCreateView.as_view(), name="api_key_create"),
    path("<str:pk>/update/", views.APIKeyUpdateView.as_view(), name="api_key_update"),
    path("<str:pk>/delete/", views.APIKeyDeleteView.as_view(), name="api_key_delete"),
]
