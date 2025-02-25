#!/bin/bash
set -e

echo "Running migrations..."
python /app/manage.py migrate

echo "Setting up static files directory..."
mkdir -p /app/staticfiles
chmod -R 777 /app/staticfiles

echo "Collecting static files..."
python /app/manage.py collectstatic --noinput --verbosity 2 || {
  echo "Static file collection failed with exit code $?"
  echo "Current directory structure:"
  find /app -type d | sort
  echo "Static files directory contents:"
  ls -la /app/staticfiles
  exit 1
}

echo "Verifying static files..."
ls -la /app/staticfiles

echo "Checking and creating superuser if necessary..."
python /app/manage.py check_create_superuser

echo "Starting application..."
exec "$@" 