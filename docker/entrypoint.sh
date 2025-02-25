#!/bin/bash
set -e

echo "Running migrations..."
python /app/manage.py migrate

echo "Collecting static files..."
python /app/manage.py collectstatic --noinput --verbosity 2 || {
  echo "Static file collection failed with exit code $?"
  echo "Current directory structure:"
  find /app -type d | sort
  exit 1
}

echo "Checking and creating superuser if necessary..."
python /app/manage.py check_create_superuser

echo "Starting application..."
exec "$@" 