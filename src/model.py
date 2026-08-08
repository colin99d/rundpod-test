import logging
import os

import torch
from diffusers import AutoPipelineForText2Image
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Defaults to sd-turbo, a tiny 100 MB model that runs on any GPU and
# generates in 1–4 steps. Set MODEL_ID to switch to a larger model
# (e.g. black-forest-labs/FLUX.1-dev) when a bigger GPU is available.
#
# On Apple Silicon sd-turbo runs on MPS directly; on CPU it runs in
# RAM. The model is small enough that no offloading is needed anywhere.


def _resolve_config() -> dict:
    """Return model + generation settings appropriate for the current hardware."""
    if torch.cuda.is_available():
        return {
            "model_id": os.environ.get("MODEL_ID", "stabilityai/sd-turbo"),
            "dtype": torch.bfloat16,
            "num_inference_steps": int(os.environ.get("NUM_STEPS", "4")),
            "size": int(os.environ.get("IMAGE_SIZE", "512")),
            "guidance_scale": 0.0,
            "max_sequence_length": None,
            "device": "cuda",
        }

    return {
        "model_id": os.environ.get("LOCAL_MODEL_ID", "stabilityai/sd-turbo"),
        "dtype": torch.float16 if torch.backends.mps.is_available() else torch.bfloat16,
        "num_inference_steps": int(os.environ.get("LOCAL_STEPS", "4")),
        "size": int(os.environ.get("LOCAL_SIZE", "512")),
        "guidance_scale": 0.0,
        "max_sequence_length": None,
        "device": "mps" if torch.backends.mps.is_available() else "cpu",
    }


CONFIG = _resolve_config()

logger.info(
    "Loading %s on %s (dtype=%s, steps=%s, size=%s)",
    CONFIG["model_id"],
    CONFIG["device"],
    CONFIG["dtype"],
    CONFIG["num_inference_steps"],
    CONFIG["size"],
)

pipe = AutoPipelineForText2Image.from_pretrained(
    CONFIG["model_id"], torch_dtype=CONFIG["dtype"]
).to(CONFIG["device"])


def generate_image(prompt: str) -> Image.Image:
    kwargs = {
        "prompt": prompt,
        "height": CONFIG["size"],
        "width": CONFIG["size"],
        "num_inference_steps": CONFIG["num_inference_steps"],
        "guidance_scale": CONFIG["guidance_scale"],
        "generator": torch.Generator(CONFIG["device"]).manual_seed(0),
    }
    if CONFIG["max_sequence_length"] is not None:
        kwargs["max_sequence_length"] = CONFIG["max_sequence_length"]

    return pipe(**kwargs).images[0]
