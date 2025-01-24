import asyncio
import logging

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class TaskTracker:
    def __init__(self):
        self.active_tasks = 0
        self.lock = asyncio.Lock()

    async def add_task(self):
        async with self.lock:
            self.active_tasks += 1

    async def remove_task(self):
        async with self.lock:
            self.active_tasks -= 1

    async def wait_for_tasks(self, timeout: float):
        async def wait():
            while self.active_tasks > 0:
                await asyncio.sleep(0.1)

        try:
            await asyncio.wait_for(wait(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning(f"Timeout waiting for {self.active_tasks} tasks to complete")


class TaskTrackerMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        await request.app.state.task_tracker.add_task()
        try:
            response = await call_next(request)
            return response
        finally:
            await request.app.state.task_tracker.remove_task()

