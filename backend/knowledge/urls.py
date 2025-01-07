from django.urls import path
from . import views

urlpatterns = [
    path('<str:organization_pk>/', views.KnowledgeBaseView.as_view(), name='knowledge_list'),
    path('<str:organization_pk>/<str:pk>/', views.PageDetailView.as_view(), name='page_detail'),
    path('<str:organization_pk>/<str:pk>/update/', views.PageUpdateView.as_view(), name='page_update'),
    path('<str:organization_pk>/<str:pk>/delete/', views.PageDeleteView.as_view(), name='page_delete'),
]
