from rest_framework.routers import DefaultRouter
from knowledge import views as knowledge_views

router = DefaultRouter()

router.register(r"knowledge", knowledge_views.PageViewSet, basename="api-knowledge")
