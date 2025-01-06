from django import forms
from .models import Agent


class AgentForm(forms.ModelForm):
    class Meta:
        model = Agent
        fields = ['name']
        widgets = {
            'name': forms.TextInput(attrs={
                'class': 'input input-bordered w-full',
                'placeholder': 'Enter agent name'
            })
        }
