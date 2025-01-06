from django.db.models import QuerySet
from rest_framework import viewsets
from rest_framework.permissions import IsAuthenticated
from hebo_organizations.models import OrganizationUser
from .models import APIKey
from .serializers import APIKeySerializer


class APIKeyViewSet(viewsets.ModelViewSet):
    serializer_class = APIKeySerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self) -> QuerySet[APIKey]:
        user = self.request.user
        if user.is_authenticated:
            org_user = OrganizationUser.objects.filter(user=user).first()
            if org_user:
                return APIKey.objects.filter(organization=org_user.organization)
        return APIKey.objects.none()

    def perform_create(self, serializer):
        if self.request.user.is_authenticated:
            org_user = OrganizationUser.objects.filter(user=self.request.user).first()
            if org_user:
                serializer.save(organization=org_user.organization)
