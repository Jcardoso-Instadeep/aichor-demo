# syntax=docker/dockerfile:1.7
# torch + CUDA 12.6 are pre-baked into this base (cached, shared base layer that
# is never re-exported). CUDA 12.6 keeps sm_60/61/75 kernels, so it runs on T4,
# P4 and P100. Only the thin extras + code layers churn on rebuilds.
FROM pytorch/pytorch:2.12.0-cuda12.6-cudnn9-runtime

# uv: fast installer (static binary from the official image).
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# The base's system python3.12 (where torch lives) is PEP-668 "externally
# managed"; let uv install the extras into that same interpreter.
ENV UV_BREAK_SYSTEM_PACKAGES=1

# Cache-mount uv's store so repeat builds reuse downloaded wheels; bind-mount
# pyproject so it doesn't bake into a layer.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv pip install --system -r pyproject.toml

WORKDIR /app
# Code last, as thin layers: editing these doesn't touch the big torch/extras
# layers, so rebuilds only re-export these.
COPY ./src ./src
COPY main.py .
COPY nn_pipeline.py .
