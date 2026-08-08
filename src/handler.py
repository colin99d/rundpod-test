from __future__ import annotations

import io
import logging
import os
import uuid

import boto3
import runpod
from pydantic import BaseModel, ValidationError

from model import generate_image

logger = logging.getLogger(__name__)

S3_BUCKET = os.environ["S3_BUCKET"]
S3_REGION = os.environ.get("S3_REGION", "us-east-1")
S3_ENDPOINT_URL = os.environ.get("S3_ENDPOINT_URL")
S3_PUBLIC_BASE_URL = os.environ.get("S3_PUBLIC_BASE_URL", "")

_s3_client = None


def _get_s3_client() -> boto3.client:
    global _s3_client
    if _s3_client is None:
        kwargs: dict = {"region_name": S3_REGION}
        if S3_ENDPOINT_URL:
            kwargs["endpoint_url"] = S3_ENDPOINT_URL
        _s3_client = boto3.client("s3", **kwargs)
    return _s3_client


class HandlerInput(BaseModel):
    prompt: str


def handler(event: dict[str, dict[str, str]]) -> dict:
    """Process incoming RunPod Serverless requests.

    Args:
        event: Contains the input data and request metadata.

    Returns:
        dict with the public S3 URL of the generated image.
    """
    logger.info("Worker Start")

    try:
        data = HandlerInput.model_validate(event.get("input"))
    except ValidationError as exc:
        first_error = exc.errors()[0]
        field = ".".join(str(loc) for loc in first_error["loc"]) or "input"
        logger.error("Invalid input: %s", exc.errors())
        return {
            "status_code": 422,
            "error": f"Invalid input: {field}: {first_error['msg']}",
        }

    logger.info(f"Received prompt: {data.prompt}")

    image = generate_image(data.prompt)
    return _upload_to_s3(image)


def _upload_to_s3(image) -> dict:
    """Upload a PIL Image to a public S3 bucket and return its URL.

    The bucket must have a public-read policy or block-public-access disabled
    so the returned URL is accessible without credentials.
    """
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)

    key = f"generated/{uuid.uuid4().hex}.png"
    client = _get_s3_client()
    client.upload_fileobj(buf, S3_BUCKET, key, ExtraArgs={"ContentType": "image/png"})

    base_url = (
        S3_PUBLIC_BASE_URL.rstrip("/")
        if S3_PUBLIC_BASE_URL
        else f"https://{S3_BUCKET}.s3.{S3_REGION}.amazonaws.com"
    )
    if S3_ENDPOINT_URL and not S3_PUBLIC_BASE_URL:
        base_url = S3_ENDPOINT_URL.rstrip("/") + f"/{S3_BUCKET}"

    url = f"{base_url}/{key}"
    logger.info(f"Uploaded image to {url}")
    return {"image_url": url}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
