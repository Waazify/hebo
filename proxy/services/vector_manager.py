import asyncpg
import logging

from fastapi import HTTPException

from db.database import DB
from db.vectorstore import VectorStore
from schemas.knowledge import CreateVectorRequest, CreateVectorResponse, Vector
from .retriever import Retriever

logger = logging.getLogger(__name__)


class VectorManager:
    def __init__(
        self,
        conn: asyncpg.Connection,
    ):
        self.db: DB = DB(conn)
        self.vectorstore: VectorStore = VectorStore(conn)

    async def create_vector(self, request: CreateVectorRequest, organization_id: str):
        try:
            agent_settings = await self.db.get_agent_settings(
                request.agent_version, organization_id
            )
            if not agent_settings:
                logger.error(
                    f"Agent settings not found for version: {request.agent_version}"
                )
                raise HTTPException(status_code=404, detail="Agent settings not found")

            if not agent_settings.embeddings:
                logger.error(
                    f"Embeddings configuration not found for agent: {request.agent_version}"
                )
                raise HTTPException(status_code=404, detail="Embeddings not found")

            retriever = Retriever(
                vector_store=self.vectorstore, agent_settings=agent_settings
            )

            logger.info(f"Embedding content for part_id: {request.part_id}")
            vector = await retriever.embed_content(request.content)

            vector_obj = Vector(
                part_id=request.part_id,
                content=request.content,
                embedding_model=request.embedding_model,
                vector=vector,
                metadata=request.metadata,
            )

            await self.vectorstore.store_vector(vector_obj)
            return CreateVectorResponse(**vector_obj.model_dump())

        except Exception as e:
            logger.error(f"Error creating vector: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
