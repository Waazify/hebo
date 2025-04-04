#!/usr/bin/env bash
set -e

echo "Running migrations..."
python /app/manage.py migrate

echo "Checking and creating superuser if necessary..."
python /app/manage.py check_create_superuser

echo "Starting application..."
exec "$@"
