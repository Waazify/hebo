from langchain_aws import ChatBedrockConverse

PROVIDER_MAP = {
    "aws": ChatBedrockConverse,
}

def init_llm(client, model_id, temperature=1, max_tokens=512):
    return PROVIDER_MAP[model_id](
        client=client,
        model=model_id,
        temperature=temperature,
        max_tokens=max_tokens,
    )
