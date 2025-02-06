import asyncio
import logging
import os
import signal
import sys
from contextlib import asynccontextmanager

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import asyncpg
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from auth.middleware import APIKeyMiddleware
from config import settings
from db.database import wait_for_database_connection
from services import HEBO
from services.middleware import TaskTracker, TaskTrackerMiddleware
from schemas.threads import (
    AddMessageRequest,
    AddMessageResponse,
    CreateThreadRequest,
    CreateThreadResponse,
    CloseThreadResponse,
    RunRequest,
)
from services.thread_manager import ThreadManager

from __version__ import __version__


logging.basicConfig(
    level=settings.LOG_LEVEL,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def graceful_shutdown(app: FastAPI):
    logger.info("Initiating graceful shutdown...")
    logger.info("Ensuring all pending conversations are completed...")
    await app.state.task_tracker.wait_for_tasks(timeout=120)
    logger.info("Closing database connections...")
    await app.state.db_pool.close()
    logger.info("Shutdown complete")


def handle_shutdown_signal(sig, frame):
    logger.info(f"Received shutdown signal: {sig}")
    global app
    asyncio.create_task(graceful_shutdown(app))


async def create_db_pool():
    db_pool = await asyncpg.create_pool(
        user=settings.DB_USER,
        password=settings.DB_PASS,
        database=settings.DB_NAME,
        host=settings.DB_HOST,
        port=settings.DB_PORT,
    )
    if db_pool:
        async with db_pool.acquire() as db_conn:
            await wait_for_database_connection(db_conn)
        return db_pool

    raise Exception("Failed to create database pool")


# This is applicable for expensive operations such as db connection pool, ML models..
# that needed to be loaded once and shared between requests
# refer to https://fastapi.tiangolo.com/advanced/events/#async-context-manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load shared data
    app.state.db_pool = await create_db_pool()
    app.state.task_tracker = TaskTracker()

    # Set up signal handlers for graceful shutdown
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, handle_shutdown_signal)

    logger.info(HEBO)
    logger.info("Application startup complete v%s", __version__)

    try:
        yield
    finally:
        logger.info("Lifespan context manager is closing")
        await graceful_shutdown(app)


app = FastAPI(
    title="hebo - proxy",
    description="hebo - proxy",
    version=__version__,
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add API Key middleware before TaskTracker middleware
app.add_middleware(APIKeyMiddleware)
app.add_middleware(TaskTrackerMiddleware)


@app.post("/threads", response_model=CreateThreadResponse)
async def create_thread(request: CreateThreadRequest, req: Request):
    """Create a new thread"""
    organization_id = req.state.organization.id
    logger.info("Creating thread for organization %s", organization_id)

    # Create DB instance for this request
    async with app.state.db_pool.acquire() as conn:
        thread_manager = ThreadManager(conn)

        # Create thread
        thread = await thread_manager.create_thread(request, organization_id)

        return CreateThreadResponse(
            thread_id=thread.id,
            contact_name=thread.contact_name,
            contact_identifier=thread.contact_identifier,
            is_open=thread.is_open,
        )


@app.post("/threads/{thread_id}/close", response_model=CloseThreadResponse)
async def close_thread(thread_id: int, req: Request):
    """Close a thread"""
    organization_id = req.state.organization.id
    logger.info("Closing thread %s for organization %s", thread_id, organization_id)

    async with app.state.db_pool.acquire() as conn:
        thread_manager = ThreadManager(conn)

        # Close thread
        thread = await thread_manager.close_thread(thread_id, organization_id)

        if not thread or not thread.id:
            raise HTTPException(status_code=404, detail="Thread not found")

        return CloseThreadResponse(
            thread_id=thread.id,
            is_open=thread.is_open,
        )


@app.get("/threads/{thread_id}")
async def get_thread(request: Request, thread_id: int):
    organization = request.state.organization
    return {
        "message": "Welcome to Hebo Messaging Service",
        "organization": organization,
    }


@app.post("/threads/{thread_id}/messages", response_model=AddMessageResponse)
async def add_message(request: AddMessageRequest, req: Request, thread_id: int):
    organization = req.state.organization
    async with app.state.db_pool.acquire() as conn:
        thread_manager = ThreadManager(conn)
        message = await thread_manager.add_message(request, thread_id, organization.id)
        return AddMessageResponse(
            message_type=message.message_type, content=message.content
        )


@app.post("/threads/{thread_id}/run")
async def run(request: RunRequest, req: Request, thread_id: int):
    """Run the agent.

    Adding a new message to the thread or invoking a new run for the same thread will result in making
    any new message from the old run expired.
    """
    organization = req.state.organization
    async with app.state.db_pool.acquire() as conn:
        thread_manager = ThreadManager(conn)
        message_stream = thread_manager.run_thread(request, thread_id, organization.id)
    return StreamingResponse(message_stream, media_type="text/event-stream")
