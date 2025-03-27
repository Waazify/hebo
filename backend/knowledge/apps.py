import logging
from concurrent.futures import ThreadPoolExecutor

from django.apps import AppConfig
from django.conf import settings

logger = logging.getLogger(__name__)


class KnowledgeConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "knowledge"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vector_executor = None

    def ready(self):
        # Initialize the thread pool executor when the app is ready
        self.vector_executor = ThreadPoolExecutor(
            max_workers=settings.VECTOR_GENERATION_MAX_WORKERS
        )
        logger.info("Initialized vector generation thread pool")

    def __del__(self):
        # Clean up the thread pool when the app is shut down
        if self.vector_executor:
            self.vector_executor.shutdown(wait=True)
            logger.info("Shut down vector generation thread pool")
