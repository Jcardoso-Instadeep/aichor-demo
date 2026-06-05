import argparse
import math
import os
import random
import time

from src.operators.jax import jaxop
from src.operators.ray import rayop
from src.operators.pytorch import pytorchop
from src.operators.xgboost import xgboostop
from src.operators.jobset import jobsetop
import logging, sys

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger(__name__)


OPERATOR_TABLE = {
    "ray": rayop,
    "kuberay": rayop,
    "jax": jaxop,
    "pytorch": pytorchop,
    "xgboost": xgboostop,
    "jobset": jobsetop
}

def print_numbered_lines(target_mb=10):
    target_bytes = int(target_mb * 1024 * 1024)
    bytes_written = 0
    line_nr = 1

    n = 50
    # while bytes_written < target_bytes:
    while line_nr < n:
        if line_nr % 3000 == 0:
            time.sleep(0.5)
        # Construct the string
        output = f"This is line {line_nr}"
        
        # print() adds a newline (\n), so we add 1 to the byte count
        # On Windows, you might want to add 2 if using \r\n
        bytes_written += len(output.encode('utf-8')) + 1
        bytes_written += 1
        
        
        logging.info(output)
        line_nr += 1

# Per-project MLflow tracking server. The experiment talks to the public URL;
# the control-plane gateway rewrites the Host to the internal one (which the
# server allows) and the client sends no Origin, so CORS does not apply.
# Override with the MLFLOW_TRACKING_URI env var if needed.
DEFAULT_MLFLOW_URI = "https://jcardoso-mlf-6a4401804a484db8-mlflow.dev.aichor.ai"


def log_runs_to_mlflow():
    """Log two runs (different learning rates) with metric curves, so MLflow
    can chart and compare them. Metrics only -> only the tracking server is
    contacted (no artifact/GCS access needed)."""
    import mlflow

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_MLFLOW_URI)
    experiment = os.environ.get("MLFLOW_EXPERIMENT_NAME", "smoke-test")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    logging.info(f"MLflow tracking_uri={tracking_uri} experiment={experiment}")

    for lr in (0.1, 0.5):
        with mlflow.start_run(run_name=f"lr-{lr}") as run:
            mlflow.log_param("learning_rate", lr)
            mlflow.log_param("epochs", 50)
            for step in range(50):
                # Synthetic training curve shaped by lr -> two distinct lines.
                loss = math.exp(-lr * step) + random.uniform(0, 0.02)
                accuracy = max(0.0, min(1.0, 1 - loss + random.uniform(-0.01, 0.01)))
                mlflow.log_metric("loss", loss, step=step)
                mlflow.log_metric("accuracy", accuracy, step=step)
            logging.info(f"finished MLflow run lr={lr} run_id={run.info.run_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='AIchor Smoke test on any operator')
    parser.add_argument("--operator", type=str, default="jobset", choices=OPERATOR_TABLE.keys(),help="operator name")
    parser.add_argument("--sleep", type=int, default="0", help="sleep time in seconds")
    parser.add_argument("--tb-write", type=bool, default=False, help="test write to tensorboard")

    args = parser.parse_args()

    print(f"using {args.operator} operator")
    OPERATOR_TABLE[args.operator](args.tb_write)

    log_runs_to_mlflow()

    print_numbered_lines(5)

    if args.sleep > 0:
        print(f"sleeping for {args.sleep}s before exiting")
        time.sleep(args.sleep)