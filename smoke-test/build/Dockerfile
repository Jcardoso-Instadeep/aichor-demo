FROM python:3.9.9-alpine

COPY build/requirements.txt .
RUN pip install -r requirements.txt

WORKDIR /app
COPY ./src ./src
COPY main.py .