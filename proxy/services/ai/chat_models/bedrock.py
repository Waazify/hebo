import boto3
from botocore.config import Config


def get_bedrock_client(
    aws_access_key_id: str,
    aws_secret_access_key: str,
    aws_region: str,
    **kwargs,
):
    # Create a Bedrock client with cross-region inference configuration
    client = boto3.Session(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=aws_region,
        **kwargs,
    ).client(
        "bedrock-runtime",
        config=Config(retries=dict(max_attempts=3)),
        **kwargs,
    )
    return client
