import asyncpg
import json
from datetime import datetime
from typing import Any, List, Optional
import logging

from schemas.ai import VectorStoreResponse
from schemas.knowledge import ContentType, Vector


class VectorStore:
    def __init__(self, conn: asyncpg.Connection):
        self.conn: asyncpg.Connection = conn

    async def store_vector(
        self,
        vector: Vector,
    ) -> int:
        """Store a vector in the database.

        Args:
            part_id: The ID of the part
            embedding_model: The embedding model
            vector: The vector embedding
            metadata: The metadata

        Returns:
            id: The ID of the stored vector

        Raises:
            ValueError: If embedding dimensions are wrong or insert fails
        """
        logger = logging.getLogger(__name__)

        # TODO: Add support for other embeddings providers
        if len(vector.vector) != 1024:
            logger.error(f"Invalid vector dimensions: {len(vector.vector)} (expected 1024)")
            raise ValueError("Embedding must be 1024 dimensions")

        # Convert the embedding list to a string format that pgvector expects
        embedding_str = f"[{','.join(str(x) for x in vector.vector)}]"

        try:
            async with self.conn.transaction():
                logger.info(f"Beginning database transaction for part_id: {vector.part_id}")
                id = await self.conn.fetchval(
                    # TODO: we are using vector_1024 (1024 is the vector dimension) for all models, we should add support for other models
                    """
                    INSERT INTO knowledge_vectorstore (part_id, content, embedding_model, vector_1024, created_at, updated_at, metadata)
                    VALUES ($1, $2, $3, $4::vector, $5, $5, $6)
                    RETURNING id
                    """,
                    vector.part_id,
                    vector.content,
                    vector.embedding_model,
                    embedding_str,
                    datetime.now(),
                    json.dumps(vector.metadata),
                )
                if id is None:
                    logger.error(f"Failed to insert vector for part_id: {vector.part_id}")
                    raise ValueError("Failed to insert vector into database")
                return id
        except Exception as e:
            logger.error(f"Database error storing vector: {str(e)}")
            raise

    async def find_similar(
        self,
        query_embedding: List[float],
        version_id: int,
        limit: int = 3,
        content_type: Optional[ContentType] = None,
        score_threshold: Optional[float] = None,
    ) -> List[VectorStoreResponse]:
        """Find similar vectors using exact nearest neighbor search with cosine distance.

        Args:
            query_embedding: The query vector embedding
            limit: Maximum number of results to return
            version_id: The version ID to filter by in metadata
            content_type: Optional filter by part type (behaviour, scenario, example)
            score_threshold: Optional similarity threshold (cosine similarity)

        Returns:
            List of VectorStoreResponse objects
        """
        # Build query using exact nearest neighbor search
        # TODO: Add support for other vector dimensions based on embedding model
        query = """
            SELECT
                vs.id,
                vs.content,
                vs.metadata,
                1 - (vs.vector_1024 <=> $1::vector) as similarity
            FROM knowledge_vectorstore vs
            """

        embedding_str = f"[{','.join(str(x) for x in query_embedding)}]"
        params: List[Any] = [embedding_str]

        conditions = []
        # Add version_id filter to match vector ID in metadata
        conditions.append(f"vs.metadata->>'version_id' = ${len(params) + 1}")
        params.append(str(version_id))

        if content_type:
            conditions.append(f"vs.metadata->>'content_type' = ${len(params) + 1}")
            params.append(content_type.value)
        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += f" ORDER BY vs.vector_1024 <=> $1::vector LIMIT ${len(params) + 1}"
        params.append(limit)

        if score_threshold is not None:
            query = f"SELECT * FROM ({query}) AS subquery WHERE similarity >= ${len(params) + 1}"
            params.append(score_threshold)

        rows = await self.conn.fetch(query, *params)

        return [
            VectorStoreResponse(
                id=row["id"],
                source=row["content"],
                similarity=row["similarity"],
                metadata={**json.loads(row["metadata"])},
            )
            for row in rows
        ]

    async def find_by_id(self, id: int) -> Optional[VectorStoreResponse]:
        """Retrieve a vector by ID."""
        query = """
            SELECT
                vs.id,
                p.content_type,
                p.start_line,
                p.end_line,
                pg.content as page_content,
                vs.vector_1024::text as vector,
                vs.metadata
            FROM knowledge_vectorstore vs
            JOIN knowledge_part p ON vs.part_id = p.id
            JOIN knowledge_page pg ON p.page_id = pg.id
            WHERE vs.id = $1
        """

        row = await self.conn.fetchrow(query, id)
        if row:
            return VectorStoreResponse(
                id=row["id"],
                source=self._extract_content(
                    row["page_content"], row["start_line"], row["end_line"]
                ),
                metadata={"content_type": row["content_type"], **row["metadata"]},
            )
        return None

    async def get_all_records(self) -> List[VectorStoreResponse]:
        """Fetch all records from the vectorstore."""
        query = """
            SELECT
                vs.id,
                p.content_type,
                p.start_line,
                p.end_line,
                pg.content as page_content,
                vs.metadata
            FROM knowledge_vectorstore vs
            JOIN knowledge_part p ON vs.part_id = p.id
            JOIN knowledge_page pg ON p.page_id = pg.id
        """

        rows = await self.conn.fetch(query)
        return [
            VectorStoreResponse(
                id=row["id"],
                source=self._extract_content(
                    row["page_content"], row["start_line"], row["end_line"]
                ),
                metadata={"content_type": row["content_type"], **row["metadata"]},
            )
            for row in rows
        ]

    @staticmethod
    def _extract_content(content: str, start_line: int, end_line: int) -> str:
        """Extract the relevant content fragment from the page content."""
        lines = content.splitlines()
        if start_line >= len(lines) or end_line > len(lines):
            return ""
        return "\n".join(lines[start_line:end_line])
