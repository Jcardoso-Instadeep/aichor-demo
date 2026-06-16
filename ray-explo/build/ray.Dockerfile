# syntax=docker/dockerfile:1.7
# Python 3.12 image (the pyproject pins require >=3.12). 2.51.1 matches the Ray
# version used elsewhere in the ecosystem and is supported by KubeRay v1.5.0.
FROM rayproject/ray:2.51.1-py312-cpu

# uv: fast installer (static binary from the official image).
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install the project deps plus the `torch` extra (the ray CPU base ships ray +
# numpy but no torch) into the image's python. Cache-mount uv's store so repeat
# builds reuse downloaded wheels; bind-mount pyproject so it isn't baked into a
# layer.
# The ray base runs as the non-root `ray` user (uid 1000, gid 100); the cache
# mount must be owned by it or uv can't write CACHEDIR.TAG.
RUN --mount=type=cache,target=/home/ray/.cache/uv,uid=1000,gid=100 \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv pip install --system --extra torch -r pyproject.toml

WORKDIR /app
# Code last, as thin layers: editing these doesn't touch the big deps layer.
COPY ./src ./src
COPY main.py .
COPY nn_pipeline.py .
