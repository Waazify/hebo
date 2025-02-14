#!/bin/sh

echo "Running npm install..."
npm install
echo "Running npm run build..."
npm run build

# Run database migrations
echo "Running migrations..."
python manage.py migrate

# Collect static files
echo "Collecting static files..."
python manage.py collectstatic --noinput

# Check for superuser creation if credentials are provided
if [ -n "$DJANGO_SUPERUSER_USERNAME" ] && [ -n "$DJANGO_SUPERUSER_EMAIL" ] && [ -n "$DJANGO_SUPERUSER_PASSWORD" ]; then
    echo "Checking if superuser exists..."
    python manage.py shell -c "from django.contrib.auth import get_user_model; import sys; User = get_user_model(); sys.exit(0) if User.objects.filter(is_superuser=True).exists() else sys.exit(1)"
    if [ $? -ne 0 ]; then
        echo "No superuser found. Creating superuser..."
        python manage.py shell -c "from django.contrib.auth import get_user_model; User = get_user_model(); User.objects.create_superuser(username='$DJANGO_SUPERUSER_USERNAME', email='$DJANGO_SUPERUSER_EMAIL', password='$DJANGO_SUPERUSER_PASSWORD')"
    else
        echo "Superuser already exists."
    fi
else
    echo "Superuser credentials not set. Skipping creation."
fi

# Execute the command passed as arguments
echo "Starting command: $@"
exec "$@" 