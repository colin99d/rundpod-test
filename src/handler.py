import logging

import runpod
from pydantic import BaseModel, ValidationError

from model import generate_image

logger = logging.getLogger(__name__)


class HandlerInput(BaseModel):
    prompt: str


def handler(event):
    #   This function processes incoming requests to your Serverless endpoint.
    #
    #    Args:
    #        event (dict): Contains the input data and request metadata
    #
    #    Returns:
    #       Any: The result to be returned to the client

    logger.info("Worker Start")

    # Validate and unwrap the input with Pydantic
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

    return generate_image(data.prompt)


# Start the Serverless function when the script is run
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
