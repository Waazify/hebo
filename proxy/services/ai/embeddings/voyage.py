from typing import Any, List, Union

import voyageai
from PIL import Image

from services.ai.langfuse_utils import trace
from services.exceptions import EmbeddingError


class VoyageClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self._client = self._init_voyage_client()

    def _init_voyage_client(self) -> voyageai.Client:
        """Initialize the VoyageAI client with API key from settings.

        Raises:
            EmbeddingException: If client initialization fails
        """
        try:
            return voyageai.Client(api_key=self.api_key)
        except Exception as e:
            raise EmbeddingError(f"Failed to initialize VoyageAI client: {str(e)}")

    @trace()
    def get_multimodal_embeddings(
        self,
        inputs: List[List[Union[str, Image.Image]]],
        input_type: str | None = None,
        truncation: bool = True,
        **kwargs: Any,
    ) -> voyageai.object.MultimodalEmbeddingsObject:  # type: ignore
        """Get multimodal embeddings from VoyageAI.

        Args:
            inputs: List of inputs, where each input is a list containing text strings
                   and PIL image objects.
            session: Optional session for tracing. If not provided, a default session will be created.
            input_type: Optional type of input ('query' or 'document').
                       Helps optimize embeddings for search use cases.
            truncation: Whether to truncate inputs that exceed the model's context length.

        Returns:
            VoyageEmbedResponse containing embeddings and metadata.

        Raises:
            EmbeddingException: If embedding generation fails

        Example:
            inputs = [
                ["This is a banana.", PIL.Image.open("banana.jpg")],
                ["Here is an apple.", PIL.Image.open("apple.jpg")]
            ]
            embeddings = get_multimodal_embeddings(inputs)
        """
        try:
            return self._client.multimodal_embed(
                inputs=inputs,
                model="voyage-multimodal-3",
                input_type=input_type,
                truncation=truncation,
                **kwargs,
            )
        except Exception as e:
            raise EmbeddingError(f"Failed to generate embeddings: {str(e)}")
