from rest_framework import serializers
from .models import APIKey


class APIKeySerializer(serializers.ModelSerializer):
    key = serializers.CharField(read_only=True)

    class Meta:
        model = APIKey
        fields = ["id", "name", "key", "created_at", "last_used_at", "is_active"]
        read_only_fields = ["key", "created_at", "last_used_at"]
