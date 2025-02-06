from django.contrib.auth.mixins import LoginRequiredMixin
from django.http import JsonResponse
from django.views.generic import View
from rest_framework import status
from rest_framework.generics import UpdateAPIView, DestroyAPIView
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from core.mixins import OrganizationPermissionMixin
from .models import APIKey
from .serializers import APIKeySerializer


class APIKeyListView(LoginRequiredMixin, OrganizationPermissionMixin, View):
    def get(self, request, *args, **kwargs):
        api_keys = APIKey.objects.filter(organization=self.organization)
        serializer = APIKeySerializer(api_keys, many=True)
        return JsonResponse({"api_keys": serializer.data})


class APIKeyCreateView(LoginRequiredMixin, OrganizationPermissionMixin, View):
    def post(self, request, *args, **kwargs):
        # Generate a default name like "API Key 1", "API Key 2", etc.
        existing_count = APIKey.objects.filter(organization=self.organization).count()
        default_name = f"API Key {existing_count + 1}"

        api_key = APIKey.objects.create(
            organization=self.organization, name=default_name
        )

        serializer = APIKeySerializer(api_key)
        return JsonResponse(serializer.data, status=201)


class APIKeyUpdateView(LoginRequiredMixin, OrganizationPermissionMixin, UpdateAPIView):
    queryset = APIKey.objects.all()
    serializer_class = APIKeySerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return APIKey.objects.filter(organization=self.organization)

    def update(self, request, *args, **kwargs):
        instance = self.get_object()
        serializer = self.get_serializer(instance, data=request.data, partial=True)
        serializer.is_valid(raise_exception=True)
        self.perform_update(serializer)
        return Response(serializer.data)


class APIKeyDeleteView(LoginRequiredMixin, OrganizationPermissionMixin, DestroyAPIView):
    queryset = APIKey.objects.all()
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return APIKey.objects.filter(organization=self.organization)

    def destroy(self, request, *args, **kwargs):
        instance = self.get_object()
        self.perform_destroy(instance)
        return Response(status=status.HTTP_204_NO_CONTENT)
