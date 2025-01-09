from django.urls import path
from . import views

urlpatterns = [
    path("", views.KnowledgeBaseView.as_view(), name="knowledge_list"),
    path("<int:pk>/", views.PageDetailView.as_view(), name="page_detail"),
    path("<int:pk>/update/", views.PageUpdateView.as_view(), name="page_update"),
    path("<int:pk>/delete/", views.PageDeleteView.as_view(), name="page_delete"),
]
