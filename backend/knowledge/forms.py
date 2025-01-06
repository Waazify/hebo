from django import forms
from .models import Page


class PageForm(forms.ModelForm):
    class Meta:
        model = Page
        fields = ["title", "content"]

    def __init__(self, *args, **kwargs):
        self.organization = kwargs.pop("organization", None)
        self.user = kwargs.pop("user", None)
        super().__init__(*args, **kwargs)
