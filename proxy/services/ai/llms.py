from langchain_aws import ChatBedrockConverse

PROVIDER_MAP = {
    "arn:aws:bedrock:us-west-2:864981741310:inference-profile/us.anthropic.claude-3-5-sonnet-20240620-v1:0": ChatBedrockConverse,
}


def init_llm(client, model_name, temperature=1, max_tokens=512):
    return PROVIDER_MAP[model_name](
        client=client,
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
    )
