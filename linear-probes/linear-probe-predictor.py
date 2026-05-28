import argparse
import json
from pathlib import Path
import numpy as np 
import torch
import time
from utils import choose_device, compute_metrics

from torch.utils.data import TensorDataset, DataLoader

timestr = time.strftime("%Y%m%d-%H%M%S")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=Path("linear-probes/features/modernbert_wifce_word_features_train"),
        help="Feature directory for training data.",
    )
    parser.add_argument(
        "--dev-dir",
        type=Path,
        default=Path("linear-probes/features/modernbert_wi_dev"),
        help="Feature directory for dev data.",
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=Path(f"linear-probes/experiments/linear-probes-{timestr}"),
        help="Directory to save probes and logs.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Maximum number of epochs per layer.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Stop a layer after this many epochs without dev improvement.",
    )
    parser.add_argument(
        "--selection-metric",
        choices=["accuracy", "recall", "precision", "f1", "f0.5"],
        default="f1",
        help="Dev metric used to select checkpoints and thresholds.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Probe learning rate.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Training batch size.",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=2048,
        help="Dev evaluation batch size.",
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.1, 0.3, 0.5, 0.7, 0.9],
        help="Positive-class probability thresholds to try on dev data.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device to use: auto, cpu, cuda, or mps.",
    )
    return parser.parse_args()


def json_safe_metrics(metrics):
    return {name: float(value) for name, value in metrics.items()}


def write_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


args = parse_args()
device = choose_device(args.device)
print(f"device={device}")

train_dir = args.train_dir
dev_dir = args.dev_dir
experiment_dir = args.experiment_dir
experiment_dir.mkdir(parents=True, exist_ok=True)
print(f"experiment_dir={experiment_dir}")

metadata = torch.load(train_dir / "metadata.pt")
y_np = np.load(train_dir / "labels.npy", mmap_mode="r")

num_hidden_states = metadata["num_hidden_states"]
hidden_size = metadata["hidden_size"]
print(f"num_hidden_states={num_hidden_states} hidden_size={hidden_size}")

y_dev = torch.from_numpy(
    np.array(np.load(dev_dir / "labels.npy", mmap_mode="r"))
).long()
print(f"loaded dev labels shape={tuple(y_dev.shape)}")


def get_probs_and_labels(probe, loader, device):
    all_probs = []
    all_labels = []

    probe.eval()

    with torch.no_grad():
        for batch_X, batch_y in loader:
            batch_X = batch_X.float().to(device)

            logits = probe(batch_X)
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu()

            all_probs.append(probs)
            all_labels.append(batch_y.cpu())

    return torch.cat(all_probs), torch.cat(all_labels)


## PROBE TRAINING

learning_rate = args.learning_rate
batch_size = args.batch_size
eval_batch_size = args.eval_batch_size
epochs = args.epochs
patience = args.patience
selection_metric = args.selection_metric
thresholds = np.array(args.thresholds)

LOG_FILE = experiment_dir / "experiment_log.md"
BEST_THRESHOLDS_FILE = experiment_dir / "best_thresholds.json"
CONFIG_FILE = experiment_dir / "config.json"
best_by_layer = {}

write_json(
    CONFIG_FILE,
    {
        "date": timestr,
        "train_dir": str(train_dir),
        "dev_dir": str(dev_dir),
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "eval_batch_size": eval_batch_size,
        "epochs": epochs,
        "patience": patience,
        "selection_metric": selection_metric,
        "thresholds": thresholds.tolist(),
        "device": str(device),
    },
)

content = f"""# Experiment Log

**Date:** {timestr}

## Parameters

| Learning Rate | Batch size | Eval batch size | Epochs | Patience | Selection metric | Thresholds |
|---------------|------------|-----------------|--------|----------|------------------|------------|
| {learning_rate} | {batch_size} | {eval_batch_size} | {epochs} | {patience} | {selection_metric} | {thresholds.tolist()} |

## Training

| Layer | Epoch | Loss | Threshold | Accuracy | Recall | Precision | F1 | F05 |
|-------|-------|------|-----------|----------|--------|-----------|----|-----|
"""

with open(LOG_FILE, "w") as f:
    f.write(content)
print(f"logging to {LOG_FILE}")

for layer_idx in range(num_hidden_states):
    print(f"layer {layer_idx}/{num_hidden_states - 1}: loading features")

    X_np = np.load(train_dir / f"layer_{layer_idx:02d}.npy", mmap_mode="r")
    X = torch.from_numpy(np.array(X_np)).float()
    y = torch.from_numpy(np.array(y_np)).long()

    X_dev_np = np.load(dev_dir / f"layer_{layer_idx:02d}.npy", mmap_mode="r")
    X_dev = torch.from_numpy(np.array(X_dev_np)).float()
    dev_dataset = TensorDataset(X_dev, y_dev)
    dev_loader = DataLoader(dev_dataset, batch_size=eval_batch_size)
    print(
        f"layer {layer_idx}: train_shape={tuple(X.shape)} "
        f"dev_shape={tuple(X_dev.shape)}"
    )

    probe = torch.nn.Linear(hidden_size, 2).to(device)

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size = batch_size, shuffle=True)

    class_counts = torch.bincount(y)
    class_weights = len(y) / (len(class_counts) * class_counts)

    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.AdamW(probe.parameters(), lr=learning_rate)

    best_score = -1
    probe_path = experiment_dir / f"probe_layer_{layer_idx:02}.pt"
    epochs_without_improvement = 0

    for epoch in range(epochs):
        print(f"layer {layer_idx}: epoch {epoch + 1}/{epochs} training")
        probe.train()
        total_loss = 0
        total_examples = 0

        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            logits = probe(batch_X)
            loss = loss_fn(logits, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_X.shape[0]
            total_examples += batch_X.shape[0]

        avg_loss = total_loss / total_examples
        print(
            f"layer {layer_idx}: epoch {epoch + 1}/{epochs} "
            f"loss={avg_loss:.4f} evaluating"
        )
        probs, y_dev_eval = get_probs_and_labels(probe, dev_loader, device)
        epoch_best_score = -1
        epoch_best_threshold = None
        epoch_improved = False

        for threshold in thresholds:
            preds = (probs >= threshold).long()
            metrics = compute_metrics(y_true=y_dev_eval, y_pred=preds)

            row = (f"| {layer_idx} | {epoch + 1} | "
                   f"{avg_loss:.4f} | "
                   f"{threshold:.2f} | "
                   f"{metrics['accuracy']:.4f} | "
                   f"{metrics['recall']:.4f} | "
                   f"{metrics['precision']:.4f} | "
                   f"{metrics['f1']:.4f} | "
                   f"{metrics['f0.5']:.4f} |\n"
            )

            with open(LOG_FILE, "a") as f:
                f.write(row)

            score = metrics[selection_metric]

            if score > best_score:
                best_score = score
                epoch_improved = True
                torch.save(probe.state_dict(), probe_path)
                best_by_layer[str(layer_idx)] = {
                    "layer": layer_idx,
                    "epoch": epoch + 1,
                    "threshold": float(threshold),
                    "selection_metric": selection_metric,
                    "selection_score": float(score),
                    "metrics": json_safe_metrics(metrics),
                    "probe_path": str(probe_path),
                }
                write_json(BEST_THRESHOLDS_FILE, best_by_layer)
                print(
                    f"layer {layer_idx}: saved new best probe "
                    f"epoch={epoch + 1} threshold={threshold:.2f} "
                    f"{selection_metric}={best_score:.4f}"
                )

            if score > epoch_best_score:
                epoch_best_score = score
                epoch_best_threshold = threshold

        print(
            f"layer {layer_idx}: epoch {epoch + 1}/{epochs} "
            f"best_threshold={epoch_best_threshold:.2f} "
            f"best_{selection_metric}={epoch_best_score:.4f}"
        )

        if epoch_improved:
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(
                f"layer {layer_idx}: early stopping after "
                f"{epochs_without_improvement} epochs without improvement"
            )
            break

    del X
    del X_dev
    del dev_loader
    del dev_dataset
    del probe

    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()
    print(f"layer {layer_idx}: done")
