import asyncio
import logging
import sys
from datetime import datetime
from functools import wraps
from typing import List, Optional
import json

import asyncpg

from schemas.agent_settings import AgentSetting, LLMAdapter, Tool
from schemas.threads import Message, MessageType, MessageContent, Thread, Run

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
        row = await self.conn.fetchrow(query, thread_id, organization_id)
        return Thread(**row) if row else None

    @db_operation
    async def close_thread(self, thread_id: int, organization_id: str) -> bool:
        """Close a thread"""
        query = """
            UPDATE threads_thread
            SET is_open = false
            WHERE id = $1 and organization_id = $2
        """
        result = await self.conn.execute(query, thread_id, organization_id)
        return "UPDATE 1" in result

    @db_operation
    async def add_message(self, message: Message) -> int:
        """Add a message to a thread"""
        query = """
            INSERT INTO threads_message (
                thread_id, created_at, message_type, content, run_status
            ) VALUES ($1, $2, $3, $4::jsonb, $5)
            RETURNING id
        """

        # Convert content list to serializable format with enum values
        serialized_content = [
            {**content.model_dump(exclude_none=True), "type": content.type.value}
            for content in message.content
        ]

        # Convert to JSON string for asyncpg
        json_content = json.dumps(serialized_content)

        message_id = await self.conn.fetchval(
            query,
            message.thread_id,
            message.created_at,
            message.message_type.value,
            json_content,
            message.run_status.value if message.run_status else None,
        )

        if message_id is None:
            raise ValueError("Failed to add message")

        return message_id

    @db_operation
    async def remove_message(self, message_id: int, thread_id: int) -> bool:
        """Remove a message from a thread"""
        query = "DELETE FROM threads_message WHERE id = $1 and thread_id = $2"
        result = await self.conn.execute(query, message_id, thread_id)
        return "DELETE 1" in result

    @db_operation
    async def get_valid_thread_messages(
        self, thread_id: int
    ) -> Optional[List[Message]]:
        """Get all messages in a thread"""
        query = """
            SELECT id, thread_id, created_at, message_type, content
            FROM threads_message
            WHERE thread_id = $1
            AND (run_status NOT IN ('error', 'expired') OR run_status IS NULL)
            AND message_type != 'comment'
            ORDER BY created_at ASC
        """
        rows = await self.conn.fetch(query, thread_id)
        if not rows:
            return None

        return [
            Message(
                id=row["id"],
                thread_id=row["thread_id"],
                created_at=row["created_at"],
                message_type=MessageType(row["message_type"]),
                content=[
                    MessageContent(**content_item)
                    for content_item in (
                        json.loads(row["content"])
                        if isinstance(row["content"], str)
                        else row["content"]
                    )
                ],
            )
            for row in rows
        ]

    @db_operation
    async def get_agent_settings(
        self, version_slug: str, organization_id: str
    ) -> Optional[AgentSetting]:
        """Get agent settings and tools for a version using its slug"""
        # First get agent settings by joining through version slugs
        settings_query = """
            SELECT
                s.id, s.organization_id, s.version_id, s.delay, s.hide_tool_messages,
                s.core_llm_id, s.condense_llm_id, s.vision_llm_id, s.embeddings_id
            FROM agent_settings_agentsetting s
            JOIN versions_versionslug vs ON vs.version_id = s.version_id
            JOIN versions_version v ON v.id = s.version_id
            WHERE vs.slug = $1 AND s.organization_id = $2
            LIMIT 1
        """
        settings_row = await self.conn.fetchrow(
            settings_query, version_slug, organization_id
        )

        if not settings_row:
            return None

        # Get tools
        tools_query = """
            SELECT id, agent_setting_id, name, description, output_template,
                   tool_type, openapi_url, auth_token, db_connection_string, query
            FROM agent_settings_tool
            WHERE agent_setting_id = $1
        """
        tools_rows = await self.conn.fetch(tools_query, settings_row["id"])

        # Get LLM adapters
        adapters_query = """
            SELECT id, is_default, organization_id, model_type, provider,
                   api_base, name, aws_region, api_key, aws_access_key_id, aws_secret_access_key
            FROM agent_settings_llmadapter
            WHERE id = ANY($1::int[])
        """
        adapter_ids = [
            settings_row["core_llm_id"],
            settings_row["condense_llm_id"],
            settings_row["vision_llm_id"],
            settings_row["embeddings_id"],
        ]
        adapters_rows = await self.conn.fetch(adapters_query, adapter_ids)

        # Convert rows to objects
        adapters_dict = {row["id"]: LLMAdapter(**dict(row)) for row in adapters_rows}

        tools = [Tool(**dict(row)) for row in tools_rows]

        # Construct AgentSetting object
        return AgentSetting(
            id=settings_row["id"],
            organization_id=settings_row["organization_id"],
            version_id=settings_row["version_id"],
            core_llm=adapters_dict.get(settings_row["core_llm_id"]),
            condense_llm=adapters_dict.get(settings_row["condense_llm_id"]),
            vision_llm=adapters_dict.get(settings_row["vision_llm_id"]),
            embeddings=adapters_dict.get(settings_row["embeddings_id"]),
            delay=settings_row["delay"],
            hide_tool_messages=settings_row["hide_tool_messages"],
            tools=tools,
        )

    @db_operation
    async def get_behaviour_part_ids(
        self, version_id: int, organization_id: str
    ) -> List[int]:
        """Get all behaviour part ids for a version, ordered by page hierarchy and part position"""
        query = """
            WITH RECURSIVE page_hierarchy AS (
                -- Get root level pages
                SELECT id, parent_id, position,
                       ARRAY[position] as path
                FROM knowledge_page
                WHERE parent_id IS NULL
                  AND version_id = $1
                  AND organization_id = $2

                UNION ALL

                -- Get child pages
                SELECT c.id, c.parent_id, c.position,
                       ph.path || c.position as path
                FROM knowledge_page c
                JOIN page_hierarchy ph ON c.parent_id = ph.id
            )
            SELECT p.id
            FROM knowledge_part p
            JOIN knowledge_page pg ON p.page_id = pg.id
            JOIN page_hierarchy ph ON pg.id = ph.id
            WHERE pg.version_id = $1
              AND pg.organization_id = $2
              AND p.content_type = 'behaviour'
              AND p.is_valid = true
            ORDER BY ph.path, p.start_line
        """
        rows = await self.conn.fetch(query, version_id, organization_id)
        return [row["id"] for row in rows]

    @db_operation
    async def create_run(self, run: Run) -> int:
        """Create a new run and return its ID"""
        query = """
            INSERT INTO threads_run (
                thread_id, organization_id, version_id, status, created_at, updated_at
            ) VALUES ($1, $2, $3, $4, $5, $5)
            RETURNING id
        """
        run_id = await self.conn.fetchval(
            query,
            run.thread_id,
            run.organization_id,
            run.version_id,
            run.status.value,
            run.created_at,
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
            WHERE r.id = $1 AND r.organization_id = $2
        """
        row = await self.conn.fetchrow(query, run_id, organization_id)
        return Run(**row) if row else None

    @db_operation
    async def update_run_status(
        self, run_id: int, status: str, organization_id: str
    ) -> bool:
        """Update the status of a run"""
        query = (
            "UPDATE threads_run SET status = $1 WHERE id = $2 AND organization_id = $3"
        )
        result = await self.conn.execute(query, status, run_id, organization_id)
        return "UPDATE 1" in result


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
