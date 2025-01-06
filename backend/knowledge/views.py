from django.contrib.auth.mixins import LoginRequiredMixin
from django.shortcuts import redirect
from django.urls import reverse
from django.views.generic import (
    CreateView,
    DetailView,
    ListView,
    UpdateView,
    DeleteView,
)
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_protect
import json

from core.mixins import OrganizationPermissionMixin
from versions.models import Version
from .models import Page
from .forms import PageForm


class KnowledgeBaseView(LoginRequiredMixin, OrganizationPermissionMixin, ListView):
    model = Page
    template_name = "knowledge/base.html"
    context_object_name = "pages"
    ordering = ["-created_at"]

    def get_queryset(self):
        # Get the selected version ID from session
        version_id = self.request.session.get("selected_version_id")
        queryset = super().get_queryset()
        
        # Filter pages by version if we have one selected
        if version_id:
            queryset = queryset.filter(version_id=version_id)
            
        return queryset.order_by(self.ordering[0])

    def get(self, request, *args, **kwargs):
        # Get the queryset first
        queryset = self.get_queryset()
        
        # If there are no pages, redirect to page creation
        if not queryset.exists():
            return redirect(reverse('page_create', kwargs={'organization_pk': self.organization.pk}))
            
        # If we're on the base knowledge URL and there are pages, redirect to the first page
        if self.request.path == reverse('knowledge_list', kwargs={'organization_pk': self.organization.pk}):
            first_page = queryset.first()
            if first_page:
                return redirect(reverse('page_detail', kwargs={
                    'organization_pk': self.organization.pk,
                    'pk': first_page.pk
                }))
        
        return super().get(request, *args, **kwargs)


class PageDetailView(LoginRequiredMixin, OrganizationPermissionMixin, DetailView):
    model = Page
    template_name = "knowledge/page_detail.html"
    context_object_name = "page"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context["current_page"] = self.object  # type: ignore
        return context


class PageCreateView(LoginRequiredMixin, OrganizationPermissionMixin, CreateView):
    model = Page
    form_class = PageForm
    template_name = "knowledge/page_form.html"

    def get_success_url(self):
        return reverse(
            "page_detail",
            kwargs={
                "organization_pk": self.organization.pk,
                "pk": self.object.pk,  # type: ignore
            },
        )

    def get_form_kwargs(self):
        kwargs = super().get_form_kwargs()
        kwargs.update({"organization": self.organization, "user": self.request.user})
        return kwargs

    def form_valid(self, form):
        form.instance.organization = self.organization
        form.instance.created_by = self.request.user

        # Access the selected version from the session
        selected_version_id = self.request.session.get("selected_version_id")
        if selected_version_id:
            form.instance.version = Version.objects.get(id=selected_version_id)

        return super().form_valid(form)


@method_decorator(csrf_protect, name='dispatch')
class PageUpdateView(LoginRequiredMixin, OrganizationPermissionMixin, UpdateView):
    model = Page
    form_class = PageForm
    
    def post(self, request, *args, **kwargs):
        if not request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return super().post(request, *args, **kwargs)
            
        try:
            data = json.loads(request.body)
            page: Page = self.get_object()  # type: ignore
            page.title = data.get('title', '').strip()
            page.content = data.get('content', '').strip()  
            page.save()
            return JsonResponse({'status': 'success'})
        except json.JSONDecodeError:
            return JsonResponse({'status': 'error', 'message': 'Invalid JSON'}, status=400)
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)}, status=500)


class PageDeleteView(LoginRequiredMixin, OrganizationPermissionMixin, DeleteView):
    model = Page
    template_name = "knowledge/page_confirm_delete.html"

    def get_success_url(self):
        return reverse(
            "knowledge_list",
            kwargs={
                "organization_pk": self.organization.pk,
            },
        )
