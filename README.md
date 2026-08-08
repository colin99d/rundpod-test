# rundpod-test

RunPod serverless endpoint that generates images from a text prompt using
[FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev).

## RunPod (production)

The Docker image runs FLUX.1-dev at full quality on the CUDA GPUs that RunPod
provides:

```bash
# Build for RunPod's worker architecture (x86_64). On Apple Silicon this must
# be explicit — otherwise Docker produces an arm64 image that RunPod cannot run.
docker build --platform linux/amd64 -t <dockerhub-username>/rundpod-test:latest .

# Log in to the registry, then push (Docker Hub or GHCR)
docker login
docker push <dockerhub-username>/rundpod-test:latest
```

Then create a Serverless endpoint in the RunPod console and set the image to:

```
<dockerhub-username>/rundpod-test:latest
```

Note: the model weights are downloaded from Hugging Face the first time a
worker boots; they are not baked into the image.

## Local runs (Apple Silicon / CPU)

FLUX.1-dev is a 12B-parameter model (~24 GB of weights in fp16) plus a 4.7B
T5-XXL text encoder. On a Mac the model lives in unified memory, which is
shared with the OS — running FLUX.1-dev there without offloading exhausts RAM,
macOS starts swapping to disk, and the machine freezes.

`src/model.py` therefore picks its configuration from the hardware:

| Hardware            | Model (default)        | Steps | Size | CPU offload |
| ------------------- | ---------------------- | ----- | ---- | ----------- |
| CUDA (RunPod)       | `stabilityai/sd-turbo` | 4     | 512  | no          |
| Apple Silicon / CPU | `stabilityai/sd-turbo` | 4     | 512  | yes         |

Local runs always keep CPU offloading enabled, so the model never fully
resides in unified memory.
Note: the default model (`sd-turbo`) is tiny (~100 MB) and downloads in
seconds. To use a larger model like FLUX.1-dev, set `MODEL_ID` and
attach a network volume for caching the weights.

### Environment overrides

Serverless (CUDA):

- `MODEL_ID` — model repo id (default: `stabilityai/sd-turbo`)
- `NUM_STEPS` — inference steps (default: 4)
- `IMAGE_SIZE` — output size (default: 512)

Local (MPS/CPU):

- `LOCAL_MODEL_ID` — smaller model to run locally (default: sd-turbo)
- `LOCAL_STEPS` — inference steps (default: 4)
- `LOCAL_SIZE` — output size (default: 512)

### Local smoke test

```bash
python3 -c "import sys; sys.path.insert(0, 'src'); from model import generate_image; generate_image('a giraffe riding a skateboard').save('out.png')"
```
