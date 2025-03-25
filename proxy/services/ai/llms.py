import re
from langchain_aws import ChatBedrockConverse

# Define regex pattern for AWS Bedrock ARNs
BEDROCK_ARN_PATTERN = re.compile(
    r"arn:aws:bedrock:[a-z0-9-]+:\d+:inference-profile/[a-zA-Z0-9.-]+:\d+"
)


# Use a function to determine the provider based on the ARN
def get_provider(model_name):
    if BEDROCK_ARN_PATTERN.match(model_name):
        return ChatBedrockConverse
    raise ValueError(f"Unsupported model: {model_name}")


def init_llm(client, model_name, temperature=1, max_tokens=512):
    provider = get_provider(model_name)
    return provider(
        client=client,
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
    )
