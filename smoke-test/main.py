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
from nn_pipeline import run_nn_pipeline, run_model_comparison
import logging, sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

# Silence the very chatty third-party libraries (boto3/GCS upload tracing,
# urllib3 connection logs, matplotlib font manager, mlflow internals).
for _noisy in ("botocore", "boto3", "s3transfer", "urllib3",
               "matplotlib", "mlflow"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)


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
DEFAULT_MLFLOW_URI = "https://jcardoso-t4-7dfc2abc90be4ef2-mlflow.dev.aichor.ai"


def _init_mlflow():
    """Point the MLflow client at the per-project tracking server. Returns the
    imported mlflow module and the base experiment name; each demo selects its
    own sub-experiment so the runs table stays uncluttered."""
    # boto3 >= 1.36 sends CRC32 request checksums that GCS's S3-interop
    # endpoint rejects (SignatureDoesNotMatch). The manifest can't inject env
    # (JobSet spec forbids it), so neutralise it here before boto3 is created.
    os.environ.setdefault("AWS_REQUEST_CHECKSUM_CALCULATION", "when_required")
    os.environ.setdefault("AWS_RESPONSE_CHECKSUM_VALIDATION", "when_required")

    import mlflow

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_MLFLOW_URI)
    base = os.environ.get("MLFLOW_EXPERIMENT_NAME", "smoke-test-v3")
    mlflow.set_tracking_uri(tracking_uri)
    logging.info(f"MLflow tracking_uri={tracking_uri} experiment_base={base}")
    return mlflow, base


def log_runs_to_mlflow(mlflow, base):
    """Log two runs (different learning rates) with metric curves, so MLflow
    can chart and compare them. Lives in its own experiment so the comparison
    view holds only these two runs."""
    mlflow.set_experiment(f"{base}-baseline")
    for lr in (0.1, 0.5):
        with mlflow.start_run(run_name=f"lr-{lr}") as run:
            # Tags: arbitrary key/value metadata. In the runs table you can add
            # a tag as a column, sort/group by it, or filter with a search like
            #   tags.framework = "synthetic" and tags.phase = "baseline"
            mlflow.set_tags({
                "framework": "synthetic",
                "phase": "baseline",
                "model_type": "decay-curve",
                "lr_bucket": "low" if lr < 0.3 else "high",
            })
            mlflow.log_param("learning_rate", lr)
            mlflow.log_param("epochs", 50)
            for step in range(50):
                # Synthetic training curve shaped by lr -> two distinct lines.
                loss = math.exp(-lr * step) + random.uniform(0, 0.02)
                accuracy = max(0.0, min(1.0, 1 - loss + random.uniform(-0.01, 0.01)))
                mlflow.log_metric("loss", loss, step=step)
                mlflow.log_metric("accuracy", accuracy, step=step)
            logging.info(f"finished MLflow run lr={lr} run_id={run.info.run_id}")


def log_artifacts_demo(mlflow, base):
    """Exercise the artifact store (GCS bucket): matplotlib figures, a confusion
    matrix, a JSON config, a text report, a CSV file and a logged table."""
    import matplotlib
    matplotlib.use("Agg")  # headless: no display in the cluster
    import matplotlib.pyplot as plt
    import numpy as np

    mlflow.set_experiment(f"{base}-artifacts")
    with mlflow.start_run(run_name="artifacts-demo") as run:
        mlflow.set_tag("demo", "artifacts")

        # 1. A matplotlib figure logged straight from the figure object.
        steps = np.arange(50)
        losses = np.exp(-0.2 * steps) + np.random.uniform(0, 0.02, size=50)
        fig, ax = plt.subplots()
        ax.plot(steps, losses, marker=".")
        ax.set_xlabel("step"); ax.set_ylabel("loss"); ax.set_title("training loss")
        mlflow.log_figure(fig, "plots/loss_curve.png")
        plt.close(fig)

        # 2. Confusion matrix (computed with numpy, rendered with matplotlib).
        labels = ["cat", "dog", "bird"]
        cm = np.array([[50, 2, 3], [4, 45, 1], [2, 3, 48]])
        fig, ax = plt.subplots()
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(labels)), labels)
        ax.set_yticks(range(len(labels)), labels)
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
        ax.set_title("confusion matrix")
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        fig.colorbar(im, ax=ax)
        mlflow.log_figure(fig, "plots/confusion_matrix.png")
        plt.close(fig)

        # 3. A dict logged as JSON, and free text logged as a file.
        mlflow.log_dict(
            {"classes": labels, "normalization": "none", "split": "test"},
            "config.json",
        )
        mlflow.log_text(
            "Smoke-test artifact run.\nConfusion matrix + loss curve attached.\n",
            "report.txt",
        )

        # 4. An arbitrary local file uploaded into an artifact sub-path.
        csv_path = "/tmp/predictions.csv"
        with open(csv_path, "w") as f:
            f.write("id,true,pred\n")
            for i in range(10):
                f.write(f"{i},{labels[i % 3]},{labels[(i + 1) % 3]}\n")
        mlflow.log_artifact(csv_path, artifact_path="data")

        # 5. A logged table (renders in the UI; stored as JSON in artifacts).
        mlflow.log_table(
            data={
                "label": labels,
                "support": [55, 50, 53],
                "precision": [0.89, 0.90, 0.92],
            },
            artifact_file="tables/per_class_metrics.json",
        )
        logging.info(f"finished MLflow artifacts-demo run_id={run.info.run_id}")


def log_nested_runs(mlflow, base):
    """A parent run with nested child runs -> a small hyper-parameter sweep.
    In the UI the children collapse under the parent; the best result is
    promoted onto the parent run."""
    mlflow.set_experiment(f"{base}-sweep")
    with mlflow.start_run(run_name="hp-sweep") as parent:
        mlflow.set_tag("run_type", "parent")
        best = None  # (final_loss, lr, batch_size)
        for lr in (0.01, 0.1, 0.5):
            for batch_size in (16, 64):
                with mlflow.start_run(run_name=f"lr{lr}-bs{batch_size}",
                                      nested=True) as child:
                    mlflow.set_tag("run_type", "child")
                    mlflow.log_params({"learning_rate": lr,
                                       "batch_size": batch_size})
                    for step in range(30):
                        loss = math.exp(-lr * step) + random.uniform(0, 0.02)
                        mlflow.log_metric("loss", loss, step=step)
                    final_loss = loss
                    mlflow.log_metric("final_loss", final_loss)
                    if best is None or final_loss < best[0]:
                        best = (final_loss, lr, batch_size)
        # Promote the winning configuration onto the parent run.
        mlflow.log_metric("best_final_loss", best[0])
        mlflow.log_params({"best_learning_rate": best[1],
                           "best_batch_size": best[2]})
        logging.info(f"finished MLflow hp-sweep best={best} "
                     f"run_id={parent.info.run_id}")


def autolog_demo(mlflow, base):
    """Let MLflow instrument the training itself. mlflow.sklearn.autolog()
    captures the estimator's params, training metrics and the fitted model
    (uploaded to the artifact store) with no explicit log_* calls."""
    try:
        from sklearn.datasets import make_classification
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
    except ImportError:
        logging.warning("scikit-learn not installed; skipping autolog demo")
        return

    mlflow.set_experiment(f"{base}-autolog")
    mlflow.sklearn.autolog()
    # One run per seed: the seed drives the dataset, the train/test split and
    # the classifier, so each run trains a different model and the metrics
    # genuinely differ -> something real to compare in the UI. flip_y adds
    # label noise and n_informative < n_features makes the task non-trivial,
    # so accuracy isn't pinned near 1.0.
    for seed in (0, 1, 2):
        X, y = make_classification(n_samples=500, n_features=10,
                                   n_informative=5, flip_y=0.1,
                                   random_state=seed)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed)
        with mlflow.start_run(run_name=f"autolog-rf-seed{seed}") as run:
            mlflow.set_tags({"demo": "autolog", "seed": seed})
            clf = RandomForestClassifier(n_estimators=50, max_depth=5,
                                         random_state=seed)
            clf.fit(X_train, y_train)  # autolog records params + training metrics
            clf.score(X_test, y_test)  # eval metrics are logged automatically too
            logging.info(f"finished autolog seed={seed} "
                         f"run_id={run.info.run_id}")
    mlflow.sklearn.autolog(disable=True)


def run_mlflow_demos():
    """Run every MLflow tracking demo against the shared, configured client."""
    mlflow, base = _init_mlflow()
    log_runs_to_mlflow(mlflow, base)
    log_artifacts_demo(mlflow, base)
    log_nested_runs(mlflow, base)
    autolog_demo(mlflow, base)
    run_nn_pipeline(mlflow, base)
    run_model_comparison(mlflow, base)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='AIchor Smoke test on any operator')
    parser.add_argument("--operator", type=str, default="jobset", choices=OPERATOR_TABLE.keys(),help="operator name")
    parser.add_argument("--sleep", type=int, default="0", help="sleep time in seconds")
    parser.add_argument("--tb-write", type=bool, default=False, help="test write to tensorboard")

    args = parser.parse_args()

    print(f"using {args.operator} operator")
    OPERATOR_TABLE[args.operator](args.tb_write)

    run_mlflow_demos()

    print_numbered_lines(5)

    if args.sleep > 0:
        print(f"sleeping for {args.sleep}s before exiting")
        time.sleep(args.sleep)