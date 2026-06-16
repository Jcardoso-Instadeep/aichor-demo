# syntax=docker/dockerfile:1.7
FROM rayproject/ray:2.23.0-cpu

# uv: fast installer (static binary from the official image).
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install the project deps plus the `torch` extra (the ray CPU base ships ray +
# numpy but no torch) into the image's python. Cache-mount uv's store so repeat
# builds reuse downloaded wheels; bind-mount pyproject so it isn't baked into a
# layer.
RUN --mount=type=cache,target=/home/ray/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv pip install --system --extra torch -r pyproject.toml

WORKDIR /app
# Code last, as thin layers: editing these doesn't touch the big deps layer.
COPY ./src ./src
COPY main.py .
COPY nn_pipeline.py .
