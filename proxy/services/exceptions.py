class ColleagueHandoffException(Exception):  # noqa: N818
    def __init__(self, message: str):
        self.message = message


class EmbeddingError(Exception):
    """Exception raised for errors in the embedding generation process."""


class RetrievalError(Exception):
    """Base exception for retrieval related errors"""
