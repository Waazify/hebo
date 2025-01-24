import uuid

from langchain_aws import ChatBedrockConverse

PROVIDER_MAP = {
    "aws": ChatBedrockConverse,
}


def generate_id() -> uuid.UUID:
    """Generate a unique ID."""
    return uuid.uuid4()


def init_llm(client, model_id, temperature=1, max_tokens=512):
    """Create a ChatBedrockConverse instance."""
    return PROVIDER_MAP[model_id](
        client=client,
        model=model_id,
        temperature=temperature,
        max_tokens=max_tokens,
    )
