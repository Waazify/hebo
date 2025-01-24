import asyncio
import logging
import sys
from datetime import datetime
from functools import wraps
from typing import Dict, List, Optional

import asyncpg

from schemas.agent_settings import AgentSetting, Tool
from schemas.knowledge import Part
from schemas.threads import Message, Run, Thread

logger = logging.getLogger(__name__)


def db_operation(f):
    """Decorator to handle database operations and errors"""

    @wraps(f)
    async def wrapper(*args, **kwargs):
        try:
            return await f(*args, **kwargs)
        except asyncpg.PostgresError as e:
            logger.error("Database error in %s: %s", f.__name__, e)
            raise
        except Exception as e:
            logger.error("Unexpected error in %s: %s", f.__name__, e)
            raise

    return wrapper


class DB:
    def __init__(self, conn: asyncpg.Connection):
        self.conn: asyncpg.Connection = conn

    @db_operation
    async def create_thread(self, thread: Thread) -> int:
        """Create a new thread and return its ID"""
        query = """
            INSERT INTO threads_thread (
                organization_id, is_open, created_at, updated_at,
                contact_name, contact_identifier
            ) VALUES ($1, $2, $3, $3, $4, $5)
            RETURNING id
        """
        thread_id = await self.conn.fetchval(
            query,
            thread.organization_id,
            True,  # is_open
            datetime.now(),
            thread.contact_name,
            thread.contact_identifier,
        )
        if thread_id is None:
            raise ValueError("Failed to create thread")
        return thread_id

    @db_operation
    async def get_thread(
        self, thread_id: int, organization_id: str
    ) -> Optional[Thread]:
        """Get a thread by its ID"""
        query = "SELECT * FROM threads_thread WHERE id = $1 and organization_id = $2"
        return await self.conn.fetchrow(query, thread_id, organization_id)

    @db_operation
    async def close_thread(self, thread_id: int, organization_id: str) -> bool:
        """Close a thread"""
        query = """
            UPDATE threads_thread
            SET is_open = false
            WHERE id = $2 and organization_id = $3
        """
        result = await self.conn.execute(query, thread_id, organization_id)
        return "UPDATE 1" in result

    @db_operation
    async def add_message(self, message: Message) -> int:
        """Add a message to a thread"""
        query = """
            INSERT INTO threads_message (
                thread_id, created_at, message_type, content
            ) VALUES ($1, $2, $3, $4)
            RETURNING id
        """
        message_id = await self.conn.fetchval(
            query,
            message.thread_id,
            message.created_at,
            message.message_type,
            [content.model_dump() for content in message.content],
        )

        if message_id is None:
            raise ValueError("Failed to add message")

        return message_id

    @db_operation
    async def remove_message(self, message_id: int) -> bool:
        """Remove a message from a thread"""
        query = "DELETE FROM threads_message WHERE id = $1"
        result = await self.conn.execute(query, message_id)
        return "DELETE 1" in result

    @db_operation
    async def get_thread_messages(
        self, thread_id: int, organization_id: str
    ) -> List[Message]:
        """Get all messages in a thread"""
        query = """
            SELECT id, thread_id, created_at, message_type, content
            FROM threads_message
            WHERE thread_id = $1 and organization_id = $2
            ORDER BY created_at ASC
        """
        rows = await self.conn.fetch(query, thread_id, organization_id)
        return [
            Message(
                thread_id=row["thread_id"],
                created_at=row["created_at"],
                message_type=row["message_type"],
                content=row["content"],
            )
            for row in rows
        ]

    @db_operation
    async def get_agent_settings(
        self, version_id: str, organization_id: str
    ) -> Optional[Dict]:
        """Get agent settings and tools for a version"""
        # First get agent settings
        settings_query = """
            SELECT organization_id, version_id, core_llm, condense_llm,
                   embeddings, delay, hide_tool_messages
            FROM agent_settings_agentsetting
            WHERE version_id = $1 and organization_id = $2
        """
        settings_row = await self.conn.fetchrow(
            settings_query, version_id, organization_id
        )

        if not settings_row:
            return None

        # Then get associated tools
        tools_query = """
            SELECT name, description, output_template, tool_type,
                   openapi_url, auth_token, db_connection_string, query
            FROM agent_settings_tool
            WHERE agent_setting_id = $1
        """
        tools_rows = await self.conn.fetch(tools_query, settings_row["id"])

        # Construct response
        settings = AgentSetting(**dict(settings_row))
        tools = [Tool(**dict(row)) for row in tools_rows]

        return {"settings": settings, "tools": tools}

    @db_operation
    async def get_behaviour_parts(
        self, version_id: int, organization_id: str
    ) -> List[Part]:
        """Get all behaviour parts for a version"""
        query = """
            SELECT p.id, p.page_id, p.start_line, p.end_line,
                   p.content_hash, p.type, p.identifier,
                   p.is_handover, p.created_at, p.updated_at, p.is_valid
            FROM knowledge_part p
            JOIN knowledge_page pg ON p.page_id = pg.id
            WHERE pg.version_id = $1 and pg.organization_id = $2
              AND p.type = 'behaviour'
              AND p.is_valid = true
            ORDER BY pg.id, p.start_line
        """
        rows = await self.conn.fetch(query, version_id, organization_id)
        return [Part(**dict(row)) for row in rows]

    @db_operation
    async def create_run(self, run: Run) -> int:
        """Create a new run and return its ID"""
        query = """
            INSERT INTO threads_run (
                thread_id, version_id, status, created_at, updated_at
            ) VALUES ($1, $2, $3, $4, $4)
            RETURNING id
        """
        run_id = await self.conn.fetchval(
            query,
            run.thread_id,
            run.version_id,
            run.status,
            run.created_at
        )
        if run_id is None:
            raise ValueError("Failed to create run")
        return run_id

    @db_operation
    async def get_run(self, run_id: int, organization_id: str) -> Optional[Run]:
        """Get the status of a run"""
        query = """
            SELECT *
            FROM threads_run r
            JOIN threads_thread t ON r.thread_id = t.id
            WHERE r.id = $1 AND t.organization_id = $2
        """
        row = await self.conn.fetchrow(query, run_id, organization_id)
        return Run(**row) if row else None


async def wait_for_database_connection(db_conn):
    """Utility function to wait for database connection"""
    max_retries = 20
    retry_interval = 10

    for _ in range(max_retries):
        try:
            await db_conn.execute("SELECT 1")
            logger.info("Database is available!")
            return
        except (asyncpg.CannotConnectNowError, asyncpg.PostgresConnectionError):
            logger.warning(
                f"Database not available. Retrying in {retry_interval} seconds..."
            )
            await asyncio.sleep(retry_interval)

    logger.error("Max retries reached. Database is not available.")
    sys.exit(1)
