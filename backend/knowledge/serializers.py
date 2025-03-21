from rest_framework import serializers

from knowledge.models import Page


class PageSerializer(serializers.ModelSerializer):
    class Meta:
        model = Page
        fields = ["title", "content", "created_at", "updated_at", "position"]


class BulkPageSerializer(serializers.Serializer):
    """Serializer for bulk page update operation."""

    title = serializers.CharField(max_length=200)
    content = serializers.CharField()
    position = serializers.IntegerField()
