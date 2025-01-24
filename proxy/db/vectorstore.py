import asyncpg
import logging
from typing import Any, List, Optional

from schemas.ai import VectorStoreResponse
from schemas.knowledge import ContentType
logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(self, conn: asyncpg.Connection):
        self.conn: asyncpg.Connection = conn

    async def find_similar(
        self,
        query_embedding: List[float],
        limit: int = 3,
        content_type: Optional[ContentType] = None,
        score_threshold: Optional[float] = None,
    ) -> List[VectorStoreResponse]:
        """Find similar vectors using exact nearest neighbor search with cosine distance.

        Args:
            query_embedding: The query vector embedding
            limit: Maximum number of results to return
            content_type: Optional filter by part type (behaviour, scenario, example)
            score_threshold: Optional similarity threshold (cosine similarity)

        Returns:
            List of VectorStoreResponse objects
        """
        # Build query using exact nearest neighbor search
        # TODO: improve underlying data structure to make queries faster
        query = """
            SELECT
                vs.id,
                p.content_type,
                p.start_line,
                p.end_line,
                pg.content as page_content,
                vs.metadata,
                1 - (vs.vector <=> $1::vector) as similarity
            FROM knowledge_vectorstore vs
            JOIN knowledge_part p ON vs.part_id = p.id
            JOIN knowledge_page pg ON p.page_id = pg.id
            """

        embedding_str = f"[{','.join(str(x) for x in query_embedding)}]"
        params: List[Any] = [embedding_str]

        conditions = []
        if content_type:
            conditions.append(f"p.content_type = ${len(params) + 1}")
            params.append(content_type.value)
        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += f" ORDER BY vs.vector <=> $1::vector LIMIT ${len(params) + 1}"
        params.append(limit)

        if score_threshold is not None:
            query = f"SELECT * FROM ({query}) AS subquery WHERE similarity >= ${len(params) + 1}"
            params.append(score_threshold)

        rows = await self.conn.fetch(query, *params)

        return [
            VectorStoreResponse(
                id=row["id"],
                source=self._extract_content(
                    row["page_content"], row["start_line"], row["end_line"]
                ),
                similarity=row["similarity"],
                metadata={"content_type": row["content_type"], **row["metadata"]},
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
                vs.vector::text as vector,
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
