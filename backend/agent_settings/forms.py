from django import forms
from .models import LLMAdapter, AgentSetting
import logging

logger = logging.getLogger(__name__)


class LLMAdapterForm(forms.ModelForm):
    class Meta:
        model = LLMAdapter
        fields = [
            "model_type",
            "provider",
            "name",
            "api_base",
            "api_key",
            "aws_region",
            "aws_access_key_id",
            "aws_secret_access_key",
        ]
        widgets = {
            "api_key": forms.PasswordInput(render_value=True),
        }

    def __init__(self, *args, organization=None, **kwargs):
        self.organization = organization
        logger.debug("LLMAdapterForm.__init__: Received organization: %s", self.organization)
        super().__init__(*args, **kwargs)

        # Make aws fields are required only for Bedrock provider
        self.fields["aws_region"].required = False
        self.fields["aws_access_key_id"].required = False
        self.fields["aws_secret_access_key"].required = False
        # Make api_base and api_key optional
        self.fields["api_base"].required = False
        self.fields["api_key"].required = False

        # Make sure these fields are required
        self.fields["provider"].required = True
        self.fields["name"].required = True
        self.fields["model_type"].required = True

        # Add help texts
        self.fields["api_key"].widget.attrs["class"] = "input input-bordered w-full"
        self.fields["api_base"].widget.attrs["class"] = "input input-bordered w-full"
        self.fields["name"].widget.attrs["class"] = "input input-bordered w-full"
        self.fields["aws_region"].widget.attrs["class"] = "input input-bordered w-full"
        self.fields["aws_access_key_id"].widget.attrs["class"] = "input input-bordered w-full"
        self.fields["aws_secret_access_key"].widget.attrs["class"] = "input input-bordered w-full"

    def clean(self):
        cleaned_data = super().clean()

        if not self.organization:
            logger.error("LLMAdapterForm.clean: Organization is missing.")
            raise forms.ValidationError("Organization is required")

        # Set the organization on the instance during validation
        self.instance.organization = self.organization
        logger.debug("LLMAdapterForm.clean: Cleaned data: %s", cleaned_data)

        return cleaned_data

    def save(self, commit=True):
        instance = super().save(commit=False)
        # Ensure organization is set
        instance.organization = self.organization
        instance.is_default = False

        if commit:
            instance.save()
        logger.info("LLMAdapterForm.save: Saved adapter: %s", instance)
        return instance


class AgentSettingForm(forms.ModelForm):
    class Meta:
        model = AgentSetting
        fields = [
            "core_llm",
            "condense_llm",
            "embeddings",
            "delay",
            "hide_tool_messages",
        ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Make the LLM fields optional since they might have default values
        self.fields["core_llm"].required = False
        self.fields["condense_llm"].required = False
        self.fields["embeddings"].required = False

        # Add form-control classes to all fields
        for field in self.fields.values():
            if isinstance(field.widget, forms.CheckboxInput):
                field.widget.attrs["class"] = "toggle toggle-primary"
            else:
                field.widget.attrs["class"] = "select select-bordered w-full"

    def clean(self):
        cleaned_data = super().clean()

        # Handle empty string values from the form
        for field in ["core_llm", "condense_llm", "embeddings"]:
            if cleaned_data.get(field) == "":
                cleaned_data[field] = None
            elif cleaned_data.get(field) == "new":
                # Remove the "new" value as it's just for UI
                cleaned_data[field] = None

        return cleaned_data

    def form_valid(self, form):
        # Handle new adapter creation if needed
        if form.cleaned_data.get(
            "provider"
        ):  # This indicates a new adapter is being created
            adapter = LLMAdapter.objects.create(
                organization=self.organization,  # type: ignore
                is_default=False,
                model_type=form.cleaned_data["model_type"],
                provider=form.cleaned_data["provider"],
                name=form.cleaned_data["name"],
                api_base=form.cleaned_data["api_base"],
                api_key=form.cleaned_data["api_key"],
                aws_region=form.cleaned_data["aws_region"],
                aws_access_key_id=form.cleaned_data["aws_access_key_id"],
                aws_secret_access_key=form.cleaned_data["aws_secret_access_key"],
            )

            # Update the form's data based on which field triggered the new adapter
            if form.cleaned_data["model_type"] == LLMAdapter.ModelType.CHAT:
                if not form.cleaned_data.get("core_llm"):
                    form.cleaned_data["core_llm"] = adapter
                elif not form.cleaned_data.get("condense_llm"):
                    form.cleaned_data["condense_llm"] = adapter
            elif form.cleaned_data["model_type"] == LLMAdapter.ModelType.EMBEDDING:
                form.cleaned_data["embeddings"] = adapter

        return super().form_valid(form)  # type: ignore
