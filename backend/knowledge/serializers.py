from rest_framework import serializers

from knowledge.models import Page


class PageSerializer(serializers.ModelSerializer):
    class Meta:
        model = Page
        fields = ["title", "content", "created_at", "updated_at", "position"]
