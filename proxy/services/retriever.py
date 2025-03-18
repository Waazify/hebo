import logging
import json
from typing import List

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from db.vectorstore import VectorStore
from schemas.ai import Session
from schemas.agent_settings import AgentSetting
from schemas.knowledge import ContentType

# TODO: Add support for other embeddings providers
# TODO: provide a similar interface for other embeddings and chat models
from .ai.chat_models.bedrock import get_bedrock_client
from .ai.conversations import execute_condense
from .ai.embeddings.voyage import VoyageClient
from .exceptions import EmbeddingError, RetrievalError

logger = logging.getLogger(__name__)


class Retriever:
    def __init__(
        self,
        vector_store: VectorStore,
        agent_settings: AgentSetting,
    ):
        self.vector_store = vector_store
        self.agent_settings = agent_settings
        # TODO: Add support for other embeddings providers
        self.embeddings_client = VoyageClient(
            agent_settings.embeddings.api_key
            if agent_settings.embeddings and agent_settings.embeddings.api_key
            else ""
        )
        self.condense_client = get_bedrock_client(
            (
                agent_settings.condense_llm.aws_access_key_id
                if agent_settings.condense_llm
                and agent_settings.condense_llm.aws_access_key_id
                else ""
            ),
            (
                agent_settings.condense_llm.aws_secret_access_key
                if agent_settings.condense_llm
                and agent_settings.condense_llm.aws_secret_access_key
                else ""
            ),
            (
                agent_settings.condense_llm.aws_region
                if agent_settings.condense_llm
                and agent_settings.condense_llm.aws_region
                else ""
            ),
        )

    async def embed_content(self, content: str) -> List[float]:
        """Embed content using the embeddings client.

        Args:
            content: The content to embed

        Returns:
            The embedded content

        Raises:
            EmbeddingError: If there's an error during the embedding process
        """
        try:
            response = self.embeddings_client.get_multimodal_embeddings(
                inputs=[[content]],
                input_type="document",
            )
            return response.embeddings[0]
        except Exception as e:
            logger.error("Failed to generate embeddings: %s", str(e))
            raise EmbeddingError(f"Embedding generation failed: {str(e)}")

    async def get_relevant_sources(
        self,
        messages: List[BaseMessage],
        session: Session,
    ):
        """Get relevant sources based on conversation messages.

        Args:
            messages: List of conversation messages
            session: Optional database session

        Returns:
            String containing relevant sources separated by newlines

        Raises:
            RetrievalError: If there's an error during the retrieval process
        """
        try:
            # Filter out tool messages
            filtered_messages: List[BaseMessage] = [
                msg for msg in messages if isinstance(msg, (AIMessage, HumanMessage))
            ]

            if not filtered_messages:
                return ""

            query = self._reduce_to_query(
                filtered_messages, session, self.agent_settings
            )
            if not query:
                logger.warning("Query condensation returned empty result")
                return ""

            # Get embeddings for the query
            try:
                response = self.embeddings_client.get_multimodal_embeddings(
                    inputs=[[query]],
                    input_type="query",
                    session=session,
                )
                embeddings = response.embeddings[0]
            except EmbeddingError as e:
                logger.error("Failed to generate embeddings: %s", str(e))
                raise RetrievalError(f"Embedding generation failed: {str(e)}")

            # Search vector store
            try:
                knowledge_vectors = await self.vector_store.find_similar(
                    query_embedding=embeddings,
                    version_id=self.agent_settings.version_id,
                    limit=3,
                    content_type=ContentType.SCENARIO,
                    score_threshold=0.3,
                )
                logger.info(
                    "Retrieved %d sources from knowledge", len(knowledge_vectors)
                )
                for vector in knowledge_vectors:
                    logger.debug(
                        "Metadata: %s, Source: %s, Similarity Score: %f",
                        json.dumps(vector.metadata),
                        vector.source,
                        vector.similarity,
                    )

                example_vectors = await self.vector_store.find_similar(
                    query_embedding=embeddings,
                    version_id=self.agent_settings.version_id,
                    limit=5,
                    content_type=ContentType.EXAMPLE,
                    score_threshold=0.2,
                )
                logger.info("Retrieved %d sources from example", len(example_vectors))
                for vector in example_vectors:
                    logger.debug(
                        "Metadata: %s, Source: %s, Similarity Score: %f",
                        json.dumps(vector.metadata),
                        vector.source,
                        vector.similarity,
                    )
            except Exception as e:
                logger.error("Vector store search failed: %s", str(e))
                raise RetrievalError(f"Vector store search failed: {str(e)}")

            # TODO: the <examples> logic should be moved on the example generation, so that it can depend on the core llm used.
            return (
                "\n\n".join([vector.source for vector in knowledge_vectors])
                + "\n\n<examples>"
                + "".join([f"\n\n{vector.source}" for vector in example_vectors])
                + "\n\n</examples>"
            )

        except Exception as e:
            logger.error("Unexpected error during retrieval: %s", str(e))
            raise RetrievalError(f"Retrieval failed: {str(e)}")

    def _reduce_to_query(
        self,
        messages: List[BaseMessage],
        session: Session,
        agent_settings: AgentSetting,
    ) -> str:
        """Reduce conversation messages to a single query string.

        Args:
            messages: List of conversation messages
            session: Optional database session

        Returns:
            Query string

        Raises:
            RetrievalError: If query condensation fails
        """
        try:
            return execute_condense(
                self.condense_client, messages, session, agent_settings
            )
        except Exception as e:
            logger.error("Query condensation failed: %s", str(e))
            raise RetrievalError(f"Query condensation failed: {str(e)}")
