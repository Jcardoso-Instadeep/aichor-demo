FROM rayproject/ray:2.23.0-cpu

COPY build/requirements.txt .
RUN pip install -r requirements.txt

WORKDIR /app
COPY ./src ./src
COPY ./metrics_test ./metrics_test
COPY main.py .