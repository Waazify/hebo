#!/bin/bash
set -e

echo "Running migrations..."
python /app/manage.py migrate

echo "Setting up static files directory..."
mkdir -p /app/staticfiles
chmod -R 777 /app/staticfiles

echo "Collecting static files..."
python /app/manage.py collectstatic --noinput

echo "Checking and creating superuser if necessary..."
python /app/manage.py check_create_superuser

echo "Starting application..."
exec "$@" 