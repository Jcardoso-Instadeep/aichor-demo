"""Full PyTorch training pipeline logged to MLflow: train -> hyper-parameter
search -> evaluation. Imported and invoked from main.py.

Everything lands in the "<base>-nn" experiment:
  * each hyper-parameter trial is a nested child run logging per-epoch curves
    (train_loss / val_loss / val_accuracy vs. step=epoch);
  * the parent run records the search space and the winning configuration;
  * a final evaluation run scores the best model on a held-out test set, logs a
    confusion-matrix figure and the serialized model (uploaded to the artifact
    store / GCS bucket).

GPU is used automatically when available (CUDA), otherwise it falls back to CPU.
Heavy/optional imports live inside run_nn_pipeline so the rest of main.py keeps
working even if torch is missing from a given image.
"""
import itertools
import logging


def _pick_device(torch):
    """GPU if the runtime has CUDA, else CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _build_dataloaders(torch, seed, batch_size):
    """Synthetic 2-class problem split into train / val / test loaders."""
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    X, y = make_classification(n_samples=2000, n_features=20, n_informative=10,
                               n_classes=2, flip_y=0.05, random_state=seed)
    X = StandardScaler().fit_transform(X).astype("float32")
    y = y.astype("int64")

    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=0.2, random_state=seed)

    def loader(features, targets, shuffle):
        ds = torch.utils.data.TensorDataset(
            torch.from_numpy(features), torch.from_numpy(targets))
        return torch.utils.data.DataLoader(ds, batch_size=batch_size,
                                           shuffle=shuffle)

    return (loader(X_train, y_train, True),
            loader(X_val, y_val, False),
            loader(X_test, y_test, False),
            X.shape[1])


def _build_model(nn, input_dim, hidden_dim, dropout):
    """A small MLP: input -> hidden -> hidden -> 2 logits."""
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, 2),
    )


def _evaluate(torch, model, loader, device):
    """Return (avg_loss, accuracy, preds, targets) over a loader."""
    import torch.nn.functional as F

    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    preds_all, targets_all = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            total_loss += F.cross_entropy(logits, yb, reduction="sum").item()
            preds = logits.argmax(1)
            correct += (preds == yb).sum().item()
            n += yb.size(0)
            preds_all.append(preds.cpu())
            targets_all.append(yb.cpu())
    preds_all = torch.cat(preds_all).numpy()
    targets_all = torch.cat(targets_all).numpy()
    return total_loss / n, correct / n, preds_all, targets_all


def _train_trial(torch, nn, mlflow, config, data, device, epochs):
    """Train one hyper-parameter configuration, logging per-epoch curves to the
    active (nested) run. Returns the trained model and its best val accuracy."""
    import torch.nn.functional as F

    train_loader, val_loader, _, input_dim = data
    model = _build_model(nn, input_dim, config["hidden_dim"],
                         config["dropout"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    mlflow.log_params(config)
    mlflow.log_param("epochs", epochs)

    best_val_acc = 0.0
    for epoch in range(epochs):
        model.train()
        running, n_batches = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            optimizer.step()
            running += loss.item()
            n_batches += 1
        train_loss = running / n_batches
        val_loss, val_acc, _, _ = _evaluate(torch, model, val_loader, device)
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)
        mlflow.log_metric("val_accuracy", val_acc, step=epoch)
        best_val_acc = max(best_val_acc, val_acc)
    return model, best_val_acc


def _log_confusion_matrix(mlflow, preds, targets, labels):
    import matplotlib
    matplotlib.use("Agg")  # headless
    import matplotlib.pyplot as plt
    import numpy as np

    k = len(labels)
    cm = np.zeros((k, k), dtype=int)
    for t, p in zip(targets, preds):
        cm[t, p] += 1

    fig, ax = plt.subplots()
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(k), labels)
    ax.set_yticks(range(k), labels)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title("confusion matrix (test)")
    for i in range(k):
        for j in range(k):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax)
    mlflow.log_figure(fig, "plots/confusion_matrix.png")
    plt.close(fig)


def _model_signature(torch, model, sample_x, device):
    """Infer an MLflow model signature (input/output schema) from a sample
    batch, so the registered model shows its Inputs/Outputs. Returns
    (signature, input_example_as_numpy)."""
    from mlflow.models import infer_signature
    model.eval()
    with torch.no_grad():
        out = model(sample_x.to(device)).cpu().numpy()
    x_np = sample_x.cpu().numpy()
    return infer_signature(x_np, out), x_np


def run_nn_pipeline(mlflow, base, epochs=15, seed=0):
    """Train -> hyper-parameter search -> evaluation, all logged to MLflow."""
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        logging.warning("torch not installed; skipping NN pipeline")
        return

    torch.manual_seed(seed)
    device = _pick_device(torch)
    logging.info(f"NN pipeline using device={device}")

    mlflow.set_experiment(f"{base}-nn")
    data = _build_dataloaders(torch, seed=seed, batch_size=64)

    # --- hyper-parameter search: grid over lr x hidden_dim x dropout ---
    grid = {
        "lr": [1e-2, 1e-3],
        "hidden_dim": [32, 64],
        "dropout": [0.0, 0.2],
    }
    keys = list(grid)
    combos = [dict(zip(keys, values))
              for values in itertools.product(*(grid[k] for k in keys))]

    best = None  # (val_acc, config, model)
    with mlflow.start_run(run_name="nn-hp-search") as parent:
        mlflow.set_tags({"demo": "nn-pipeline", "stage": "search",
                         "device": str(device)})
        mlflow.log_param("search_space", grid)
        mlflow.log_param("n_trials", len(combos))
        for i, config in enumerate(combos):
            with mlflow.start_run(run_name=f"trial-{i}", nested=True):
                mlflow.set_tags({"stage": "trial", "device": str(device)})
                model, val_acc = _train_trial(
                    torch, nn, mlflow, config, data, device, epochs)
                mlflow.log_metric("best_val_accuracy", val_acc)
                if best is None or val_acc > best[0]:
                    best = (val_acc, config, model)
        # Promote the winning configuration onto the parent run.
        mlflow.log_metric("best_val_accuracy", best[0])
        mlflow.log_params({f"best_{k}": v for k, v in best[1].items()})
        logging.info(f"NN search best val_acc={best[0]:.4f} config={best[1]}")

    # --- final evaluation of the best model on the held-out test set ---
    best_val_acc, best_config, best_model = best
    _, _, test_loader, _ = data
    with mlflow.start_run(run_name="nn-eval") as run:
        mlflow.set_tags({"demo": "nn-pipeline", "stage": "eval",
                         "device": str(device)})
        mlflow.log_params({f"best_{k}": v for k, v in best_config.items()})
        test_loss, test_acc, preds, targets = _evaluate(
            torch, best_model, test_loader, device)
        mlflow.log_metric("test_loss", test_loss)
        mlflow.log_metric("test_accuracy", test_acc)
        _log_confusion_matrix(mlflow, preds, targets,
                              labels=["class 0", "class 1"])
        # (Model logging happens only in the model-comparison experiment.)
        logging.info(f"NN eval test_acc={test_acc:.4f} test_loss={test_loss:.4f} "
                     f"run_id={run.info.run_id}")


# ---------------------------------------------------------------------------
# Model-complexity comparison: small vs large MLP on a hard, nonlinear dataset.
# ---------------------------------------------------------------------------

def _build_mlp(nn, input_dim, hidden_dims, n_classes, dropout=0.0):
    """A configurable MLP: input -> [hidden -> ReLU -> Dropout]* -> n_classes.
    Pass hidden_dims=[16] for a tiny model or [256, 256, 128] for a deep one."""
    layers = []
    prev = input_dim
    for h in hidden_dims:
        layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
        prev = h
    layers.append(nn.Linear(prev, n_classes))
    return nn.Sequential(*layers)


def _build_teacher_dataset(torch, nn, seed, n_samples=24000, n_features=40,
                           batch_size=128, flip_frac=0.02):
    """A smooth-but-nonlinear binary task. A fixed shallow "teacher" with
    normalized weights (so the boundary is learnable, not high-frequency noise)
    produces a score; we threshold at its median for balanced labels. A large
    MLP can fit this well; a tiny MLP underfits -> a clear, *noticeable*
    accuracy gap. Fully synthetic, no download.

    (An earlier version used a deep, large-weight teacher whose function was so
    high-frequency that neither model could learn it -> both ~chance, tiny gap.)"""
    n_classes = 2
    gen = torch.Generator().manual_seed(seed)
    X = torch.randn(n_samples, n_features, generator=gen)

    # Shallow smooth teacher: x -> tanh(x W1) W2, normalized init keeps it learnable.
    hidden = 64
    with torch.no_grad():
        W1 = torch.randn(n_features, hidden, generator=gen) / (n_features ** 0.5)
        W2 = torch.randn(hidden, 1, generator=gen) / (hidden ** 0.5)
        score = (torch.tanh(X @ W1) @ W2).squeeze(1)
        y = (score > score.median()).long()  # balanced binary labels

    # Flip a small fraction of labels so the ceiling is < 100% (realistic).
    n_flip = int(flip_frac * n_samples)
    flip_idx = torch.randperm(n_samples, generator=gen)[:n_flip]
    y[flip_idx] = 1 - y[flip_idx]

    # Standardize features (already ~N(0,1), but keep it explicit/robust).
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    X = X.float()

    n_test = int(0.15 * n_samples)
    n_val = int(0.15 * n_samples)
    perm = torch.randperm(n_samples, generator=gen)
    test_i, val_i, train_i = perm[:n_test], perm[n_test:n_test + n_val], perm[n_test + n_val:]

    def loader(idx, shuffle):
        ds = torch.utils.data.TensorDataset(X[idx], y[idx])
        return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    return (loader(train_i, True), loader(val_i, False), loader(test_i, False),
            n_features, n_classes)


def _train_and_log(torch, nn, mlflow, model, train_loader, val_loader,
                   device, epochs, lr):
    """Train a pre-built model, logging per-epoch curves to the active run.
    Returns the best validation accuracy seen."""
    import torch.nn.functional as F

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_acc = 0.0
    for epoch in range(epochs):
        model.train()
        running, n_batches = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            optimizer.step()
            running += loss.item()
            n_batches += 1
        train_loss = running / n_batches
        val_loss, val_acc, _, _ = _evaluate(torch, model, val_loader, device)
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)
        mlflow.log_metric("val_accuracy", val_acc, step=epoch)
        best_val_acc = max(best_val_acc, val_acc)
    return best_val_acc


def run_model_comparison(mlflow, base, epochs=35, seed=0):
    """Train a small and a large MLP on the same hard dataset, compare them, and
    register both in the model registry with the winner aliased @champion."""
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        logging.warning("torch not installed; skipping model comparison")
        return

    torch.manual_seed(seed)
    device = _pick_device(torch)
    logging.info(f"model comparison using device={device}")

    mlflow.set_experiment(f"{base}-model-comparison")
    (train_loader, val_loader, test_loader,
     input_dim, n_classes) = _build_teacher_dataset(torch, nn, seed)
    class_names = [f"class {i}" for i in range(n_classes)]

    # Two architectures of very different capacity, same training budget.
    architectures = {
        # Deliberately under-powered: 4 hidden units can't capture the boundary.
        "small-mlp": {"hidden_dims": [4], "dropout": 0.0, "lr": 1e-3},
        # Ample capacity to fit the smooth nonlinear teacher.
        "large-mlp": {"hidden_dims": [256, 256, 128], "dropout": 0.1, "lr": 1e-3},
    }
    registered_name = "model-comparison-net"
    results = {}

    with mlflow.start_run(run_name="model-comparison") as parent:
        mlflow.set_tags({"demo": "model-comparison", "device": str(device)})
        mlflow.log_params({"input_dim": input_dim, "n_classes": n_classes,
                           "epochs": epochs, "dataset": "deep-teacher-net"})

        for name, cfg in architectures.items():
            with mlflow.start_run(run_name=name, nested=True) as child:
                mlflow.set_tags({"architecture": name, "device": str(device)})
                model = _build_mlp(nn, input_dim, cfg["hidden_dims"],
                                   n_classes, cfg["dropout"]).to(device)
                n_params = sum(p.numel() for p in model.parameters())
                mlflow.log_params({"hidden_dims": str(cfg["hidden_dims"]),
                                   "dropout": cfg["dropout"], "lr": cfg["lr"],
                                   "n_parameters": n_params})

                best_val = _train_and_log(torch, nn, mlflow, model, train_loader,
                                          val_loader, device, epochs, cfg["lr"])
                test_loss, test_acc, preds, targets = _evaluate(
                    torch, model, test_loader, device)
                mlflow.log_metric("best_val_accuracy", best_val)
                mlflow.log_metric("test_accuracy", test_acc)
                mlflow.log_metric("test_loss", test_loss)
                _log_confusion_matrix(mlflow, preds, targets, class_names)

                # Register this architecture as a new version, with an inferred
                # input/output schema.
                try:
                    sample_x = next(iter(test_loader))[0][:5]
                    sig, input_example = _model_signature(torch, model, sample_x, device)
                    mlflow.pytorch.log_model(model, name="model",
                                             registered_model_name=registered_name,
                                             signature=sig, input_example=input_example)
                except Exception as exc:  # noqa: BLE001 - best-effort
                    logging.warning(f"could not log/register model {name}: {exc}")

                results[name] = {"test_acc": test_acc, "n_params": n_params,
                                 "run_id": child.info.run_id}
                logging.info(f"{name}: params={n_params} test_acc={test_acc:.4f}")

        # Summarize the comparison on the parent run.
        winner = max(results, key=lambda k: results[k]["test_acc"])
        gap = abs(results["large-mlp"]["test_acc"] - results["small-mlp"]["test_acc"])
        mlflow.log_param("winner", winner)
        mlflow.log_metric("winner_test_accuracy", results[winner]["test_acc"])
        mlflow.log_metric("accuracy_gap", gap)
        mlflow.log_dict(results, "comparison_summary.json")

        # Point the @champion alias at the winning model version.
        try:
            from mlflow.tracking import MlflowClient
            client = MlflowClient()
            versions = client.search_model_versions(f"name='{registered_name}'")
            win_run = results[winner]["run_id"]
            win_version = next((mv.version for mv in versions
                                if mv.run_id == win_run), None)
            if win_version is not None:
                client.set_registered_model_alias(
                    registered_name, "champion", win_version)
                logging.info(f"aliased {registered_name}@champion -> "
                             f"v{win_version} ({winner})")
        except Exception as exc:  # noqa: BLE001 - best-effort
            logging.warning(f"could not set champion alias: {exc}")

        logging.info(f"model comparison winner={winner} gap={gap:.4f} "
                     f"run_id={parent.info.run_id}")


# ---------------------------------------------------------------------------
# Observability demo: (1) system metrics (CPU/GPU/mem sampled during the run)
# and (2) dataset lineage -> two runs trained on two DIFFERENT datasets, each
# recorded, so the experiment's Datasets view traces each run to its own data.
# ---------------------------------------------------------------------------

def _build_lineage_dataset(mlflow, np, name, source, n_features, ds_seed,
                           n_samples=40000, noise=0.0):
    """Generate a synthetic dataset and wrap it as an MLflow dataset for
    lineage. Distinct name/source/content -> a distinct digest in the UI."""
    rng = np.random.default_rng(ds_seed)
    X = rng.standard_normal((n_samples, n_features)).astype("float32")
    w = rng.standard_normal((n_features,)).astype("float32")
    score = X @ w
    if noise:
        score = score + rng.standard_normal(n_samples).astype("float32") * noise * score.std()
    y = (score > 0).astype("int64")

    dataset = None
    try:
        import pandas as pd
        # A small sample carries the schema; lineage needs the descriptor, not all rows.
        df = pd.DataFrame(X[:1000, :8], columns=[f"f{i}" for i in range(8)])
        df["label"] = y[:1000]
        try:
            dataset = mlflow.data.from_pandas(df, source=source, name=name,
                                              targets="label")
        except Exception:
            dataset = mlflow.data.from_pandas(df, name=name, targets="label")
    except Exception as exc:  # noqa: BLE001 - lineage is best-effort
        logging.warning(f"could not build mlflow dataset {name}: {exc}")
    return X, y, dataset


def run_observability_demo(mlflow, base, seed=0, epochs=40):
    """Two runs, each trained on a DIFFERENT dataset (recorded for lineage) with
    system-metrics logging on. The experiment's Datasets view then shows each
    run traced to its own dataset; each run's System metrics tab shows
    CPU/GPU/memory curves."""
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except ImportError:
        logging.warning("torch not installed; skipping observability demo")
        return
    import os
    import numpy as np

    # Sample frequently so even a short run captures several points
    # (default interval is 10s; each run is tens of seconds).
    os.environ.setdefault("MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL", "2")
    os.environ.setdefault("MLFLOW_SYSTEM_METRICS_SAMPLES_BEFORE_LOGGING", "1")

    torch.manual_seed(seed)
    device = _pick_device(torch)
    logging.info(f"observability demo using device={device}")
    mlflow.set_experiment(f"{base}-observability")

    # Two distinct datasets -> two runs -> lineage differs per run.
    dataset_configs = [
        {"name": "synthetic-v1", "source": "synthetic://teacher-v1",
         "n_features": 128, "ds_seed": 1, "noise": 0.0},
        {"name": "synthetic-v2", "source": "synthetic://teacher-v2",
         "n_features": 256, "ds_seed": 2, "noise": 0.15},
    ]

    for cfg in dataset_configs:
        X, y, dataset = _build_lineage_dataset(mlflow, np, **cfg)
        n_features = X.shape[1]
        Xt = torch.from_numpy(X).to(device)
        yt = torch.from_numpy(y).to(device)
        model = _build_mlp(nn, n_features, [512, 512, 256], 2, dropout=0.1).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # log_system_metrics=True -> MLflow samples CPU/GPU/memory in the background.
        with mlflow.start_run(run_name=f"train-{cfg['name']}",
                              log_system_metrics=True) as run:
            mlflow.set_tags({"demo": "observability", "dataset": cfg["name"],
                             "device": str(device)})
            mlflow.log_params({"n_features": n_features,
                               "hidden_dims": "[512, 512, 256]", "epochs": epochs})
            if dataset is not None:
                mlflow.log_input(dataset, context="training")  # dataset lineage

            batch = 2048
            n_batches = X.shape[0] // batch
            for epoch in range(epochs):
                perm = torch.randperm(X.shape[0], device=device)
                running = 0.0
                for i in range(n_batches):
                    idx = perm[i * batch:(i + 1) * batch]
                    optimizer.zero_grad()
                    loss = F.cross_entropy(model(Xt[idx]), yt[idx])
                    loss.backward()
                    optimizer.step()
                    running += loss.item()
                mlflow.log_metric("train_loss", running / n_batches, step=epoch)
            logging.info(f"observability run dataset={cfg['name']} "
                         f"run_id={run.info.run_id}")
