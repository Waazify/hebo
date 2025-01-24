import base64
import httpx

from schemas.threads import Message, MessageContentType


async def _encode_image_from_url(url: str) -> str:
    async with httpx.AsyncClient() as client:
        # Download the image
        response = await client.get(url)
        if response.status_code != 200:
            raise Exception(f"Failed to download image from {url}")

        # Encode the image
        return base64.b64encode(response.content).decode()


def _image_url_body(encoded_image: str) -> dict:
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"},
    }


def _text_body(text: str) -> dict:
    return {"type": "text", "text": text}


async def get_content_from_human_message(message: Message) -> list:
    content = message.content
    urls = [c.image_url for c in content if c.type == MessageContentType.IMAGE_URL]
    texts = [c.text for c in content if c.type == MessageContentType.TEXT]

    content = []

    for url in urls:
        if url:
            encoded_image = await _encode_image_from_url(url)
            content.append(_image_url_body(encoded_image))

    for text in texts:
        if text:
            content.append(_text_body(text))

    if not content:
        raise ValueError("Message is not human")

    return content
