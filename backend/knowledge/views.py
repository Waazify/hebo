import json
import logging
import markdown

from django.contrib.auth.mixins import LoginRequiredMixin
from django.http import JsonResponse
from django.shortcuts import redirect
from django.urls import reverse, reverse_lazy
from django.utils.decorators import method_decorator
from django.utils.safestring import mark_safe
from django.views.decorators.csrf import csrf_protect
from django.views.generic import (
    DeleteView,
    DetailView,
    ListView,
    UpdateView,
)
from django.db import transaction

from core.mixins import OrganizationPermissionMixin
from versions.models import Version
from .forms import PageForm
from .models import Page

logger = logging.getLogger(__name__)


def extract_title_and_content(content: str) -> tuple[str, str]:
    """Extract title from first line and format content."""
    lines = content.strip().split("\n", 1)

    if not lines:
        return "New Page", ""

    first_line = lines[0].strip()
    remaining_content = lines[1].strip() if len(lines) > 1 else ""

    # If first line isn't a header, make it one
    if not first_line.startswith("# "):
        title = first_line
        formatted_content = f"# {first_line}\n\n{remaining_content}"
    else:
        title = first_line[2:].strip()  # Remove '# ' prefix
        formatted_content = content

    return title, formatted_content


class KnowledgeBaseView(LoginRequiredMixin, OrganizationPermissionMixin, ListView):
    model = Page
    template_name = "knowledge/base.html"
    context_object_name = "pages"
    ordering = ["-created_at"]

    def get_queryset(self):
        version_id = self.request.session.get("selected_version_id")
        queryset = super().get_queryset()
        if version_id:
            queryset = queryset.filter(version_id=version_id)
        # Return only top-level pages ordered by position
        return queryset.filter(parent__isnull=True).order_by("position")

    def get(self, request, *args, **kwargs):
        # Get the queryset first
        queryset = self.get_queryset()

        # If there are no pages, create one and redirect to it
        if not queryset.exists():
            # Get the selected version
            version_id = request.session.get("selected_version_id")
            version = None
            if version_id:
                try:
                    version = Version.objects.get(id=version_id)
                except Version.DoesNotExist:
                    raise ValueError("Version not found")

            # Create the first page
            page = Page.objects.create(
                title="New Page",
                content="# New Page\n\nStart writing here...",
                organization=self.organization,
                version=version,
            )

            return redirect(
                reverse(
                    "page_detail",
                    kwargs={"organization_pk": self.organization.pk, "pk": page.pk},
                )
            )

        # If we're on the base knowledge URL and there are pages, redirect to the first page
        if self.request.path == reverse(
            "knowledge_list",
            kwargs={"organization_pk": self.organization.pk},
        ):
            first_page = queryset.first()
            if first_page:
                return redirect(
                    reverse(
                        "page_detail",
                        kwargs={
                            "organization_pk": self.organization.pk,
                            "pk": first_page.pk,
                        },
                    )
                )

        return super().get(request, *args, **kwargs)

    def post(self, request, *args, **kwargs):
        if request.headers.get("X-Requested-With") == "XMLHttpRequest":
            try:
                data = json.loads(request.body)

                # Handle page reordering if that's the request type
                if data.get("action") == "reorder":
                    return self.handle_reorder(request, data)

                # Existing page creation logic
                content = data.get("content", "").strip()
                title, formatted_content = extract_title_and_content(content)

                # Get version - either from request or session
                version_id = data.get("version") or request.session.get(
                    "selected_version_id"
                )

                if not version_id:
                    raise ValueError("A version must be selected to create a page")

                try:
                    version = Version.objects.get(id=version_id)
                except Version.DoesNotExist:
                    error_msg = f"Version not found: {version_id}"
                    raise ValueError(error_msg)

                # Create new page
                page = Page.objects.create(
                    title=title,
                    content=formatted_content,
                    organization=self.organization,
                    version=version,
                )

                response_data = {
                    "status": "success",
                    "redirect_url": reverse(
                        "page_detail",
                        kwargs={
                            "organization_pk": self.organization.pk,
                            "pk": page.pk,
                        },
                    ),
                }
                return JsonResponse(response_data)
            except Exception as e:
                return JsonResponse(
                    {
                        "status": "error",
                        "message": str(e),
                    },
                    status=500,
                )

        return super().post(request, *args, **kwargs)  # type: ignore

    @transaction.atomic
    def handle_reorder(self, request, data):
        try:
            page_id = data.get("page_id")
            new_parent_id = data.get("parent_id")
            new_position = data.get("position", 0)
            old_parent_id = data.get("old_parent_id")

            page = Page.objects.get(id=page_id, organization=self.organization)
            page.parent_id = new_parent_id  # type: ignore
            page.position = new_position
            page.save()

            # Reorder siblings in new and old parent containers
            Page.reorder_positions(parent_id=new_parent_id)
            if old_parent_id != new_parent_id:
                Page.reorder_positions(parent_id=old_parent_id)

            return JsonResponse({"status": "success", "message": "Page reordered"})

        except Exception as e:
            logger.exception("Reorder failed")
            return JsonResponse({"status": "error", "message": str(e)}, status=500)

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        # Add the selected version ID to the context
        context["selected_version_id"] = self.request.session.get("selected_version_id")
        return context


@method_decorator(csrf_protect, name="dispatch")
class PageDetailView(LoginRequiredMixin, OrganizationPermissionMixin, DetailView):
    model = Page
    template_name = "knowledge/page_detail.html"
    context_object_name = "page"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        md = markdown.Markdown(
            extensions=[
                "extra",
                "codehilite",
                "fenced_code",
                "tables",
                "toc",
            ]
        )
        context["page_content"] = md.convert(self.object.content)  # type: ignore
        context["raw_content"] = self.object.content  # type: ignore
        context["current_page"] = self.object  # type: ignore
        return context


@method_decorator(csrf_protect, name="dispatch")
class PageUpdateView(LoginRequiredMixin, OrganizationPermissionMixin, UpdateView):
    model = Page
    form_class = PageForm

    def post(self, request, *args, **kwargs):
        if not request.headers.get("X-Requested-With") == "XMLHttpRequest":
            return super().post(request, *args, **kwargs)

        try:
            data = json.loads(request.body)
            page: Page = self.get_object()  # type: ignore

            # Get the raw markdown content and extract title
            content = data.get("content", "").strip()
            title, formatted_content = extract_title_and_content(content)

            # Convert to HTML for the response
            md = markdown.Markdown(
                extensions=[
                    "extra",
                    "codehilite",
                    "fenced_code",
                    "tables",
                    "toc",
                ]
            )
            html_content = md.convert(formatted_content)

            # Save both title and content
            page.title = title
            page.content = formatted_content
            page.save()

            return JsonResponse(
                {
                    "status": "success",
                    "html_content": mark_safe(html_content),
                    "title": title,
                    "page_id": str(page.pk),
                }
            )
        except Exception as e:
            logger.error(f"Error updating page: {str(e)}")
            return JsonResponse({"status": "error", "message": str(e)}, status=500)


class PageDeleteView(LoginRequiredMixin, OrganizationPermissionMixin, DeleteView):
    model = Page
    template_name = "knowledge/page_confirm_delete.html"

    def get_success_url(self):
        return reverse_lazy(
            "knowledge_list",
            kwargs={"organization_pk": self.organization.pk},
        )
