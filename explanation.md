# Explanation

## What

A RunPod serverless endpoint that generates an image from a text prompt, uploads
it to S3/R2, and returns a public URL. A vanilla-HTML frontend (`index.html`)
lets anyone test it from a browser with an API key.

## Why the pieces are what they are

**Serverless, not a persistent GPU** — an idle A100 costs hundreds/month;
serverless scales to zero and you pay only for inference seconds. Cold starts
are ~30–60 s, fine for intermittent use.

**`sd-turbo` as default, FLUX as override** — FLUX.1-dev is ~24 GB and would
swap-to-death on Apple Silicon during local dev. The 100 MB `sd-turbo` runs
anywhere in seconds. Production can switch to FLUX by setting
`MODEL_ID=black-forest-labs/FLUX.1-dev`.

**Hardware auto-detection** — `model.py` picks CUDA → MPS → CPU at import time.
Separate env vars for serverless (`MODEL_ID`, `NUM_STEPS`, `IMAGE_SIZE`) vs.
local (`LOCAL_MODEL_ID`, `LOCAL_STEPS`, `LOCAL_SIZE`) prevent a MacBook from
accidentally trying to load FLUX.

**S3/R2, not inline base64** — RunPod responses cap at ~10 MB. Returning a URL
is a one-liner on the client (`img.src = url`) and avoids encoding overhead.

**Docker: Debian + uv + `linux/amd64`** — PyTorch lacks musllinux wheels so
Alpine is out. uv builds faster than pip. RunPod workers are x86_64, so Apple
Silicon builds must explicitly target that platform.

**Single-file HTML frontend** — zero deps, zero build, instant deploy. Uses
RunPod's `runsync` API with a polling fallback if the job is queued.

**Pydantic input validation** — structured errors (`422` with `loc` + `msg`)
that the frontend can display, and easy to extend with new optional params.

## Local testing

```bash
# Smoke test — just the model
python3 -c "import sys; sys.path.insert(0, 'src'); from model import generate_image; generate_image('test').save('out.png')"

# Full handler (requires S3 env vars)
python3 src/handler.py
```
