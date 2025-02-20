from django.contrib.auth.decorators import login_required
from django.shortcuts import render
from django.http import JsonResponse
from django.db import connections
from django.db.utils import OperationalError
from django.conf import settings

@login_required(login_url='account_login')
def home(request):
    return render(request, 'home.html')


def health_check(request):
    """
    Health check endpoint that verifies database connectivity
    and returns application status.
    """
    # Check database connectivity
    db_status = "healthy"
    try:
        connections['default'].ensure_connection()
    except OperationalError:
        db_status = "unhealthy"

    # Build response data
    health_data = {
        "status": "healthy" if db_status == "healthy" else "unhealthy",
        "version": settings.APP_VERSION,
        "database": db_status
    }

    status_code = 200 if db_status == "healthy" else 503
    return JsonResponse(health_data, status=status_code)
