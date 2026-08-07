# uv image that ships Python 3.12, matching requires-python in pyproject.toml
# Debian-based (not alpine): torch does not ship musllinux wheels
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

# Copy the dependency manifest first so dependency layers are cached across rebuilds
COPY pyproject.toml ./

# uv lock pins the resolution; uv sync creates the .venv and installs from the lockfile
RUN uv lock && uv sync --frozen --no-dev --no-install-project

# Copy the serverless handler
COPY src/ ./

# Start the serverless function
CMD ["uv", "run", "--no-sync", "handler.py"]
