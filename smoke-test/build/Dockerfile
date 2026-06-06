# torch + CUDA 12.6 are pre-baked into this base (cached layer, not rebuilt each
# time). CUDA 12.6 keeps sm_60/sm_61 kernels, so it runs on T4, P4 and P100.
# Only the light extras (mlflow, sklearn, matplotlib, pandas, ...) are installed
# on top, so "exporting layers" stays small and fast.
FROM pytorch/pytorch:2.12.0-cuda12.6-cudnn9-runtime

COPY build/requirements.txt .
RUN pip install -r requirements.txt

WORKDIR /app
COPY ./src ./src
COPY main.py .
COPY nn_pipeline.py .