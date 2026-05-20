from pathlib import Path
import numpy as np 
import torch
import time
from utils import choose_device, compute_metrics

from torch.utils.data import TensorDataset, DataLoader

timestr = time.strftime("%Y%m%d-%H%M%S")
device = choose_device("auto")

train_dir = Path("modernbert_word_features_train")
dev_dir = Path("modernbert_word_features_dev")
experiment_dir = Path(f"experiments/linear-probes-{timestr}")
experiment_dir.mkdir(parents=True, exist_ok=True)

metadata = torch.load(train_dir / "metadata.pt")
y_np = np.load(train_dir / "labels.npy", mmap_mode="r")

num_hidden_states = metadata["num_hidden_states"]
hidden_size = metadata["hidden_size"]

y_dev = torch.from_numpy(
    np.array(np.load(dev_dir / "labels.npy", mmap_mode="r"))
).long()


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

learning_rate = 1e-3
batch_size = 256
eval_batch_size = 2048
epochs = 10
thresholds = np.arange(0.1, 1, 0.2)

LOG_FILE = experiment_dir / "experiment_log.md"

content = f"""# Experiment Log

**Date:** {timestr}

## Parameters

| Learning Rate | Batch size | Eval batch size | Epochs | Thresholds |
|---------------|------------|-----------------|--------|------------|
| {learning_rate} | {batch_size} | {eval_batch_size} | {epochs} | {thresholds.tolist()} |

## Training

| Layer | Epoch | Loss | Threshold | Accuracy | Recall | Precision | F1 | F05 |
|-------|-------|------|-----------|----------|--------|-----------|----|-----|
"""

with open(LOG_FILE, "w") as f:
    f.write(content)

for layer_idx in range(num_hidden_states):
    print(f"training layer {layer_idx} probe")

    X_np = np.load(train_dir / f"layer_{layer_idx:02d}.npy", mmap_mode="r")
    X = torch.from_numpy(np.array(X_np)).float()
    y = torch.from_numpy(np.array(y_np)).long()

    X_dev_np = np.load(dev_dir / f"layer_{layer_idx:02d}.npy", mmap_mode="r")
    X_dev = torch.from_numpy(X_dev_np)
    dev_dataset = TensorDataset(X_dev, y_dev)
    dev_loader = DataLoader(dev_dataset, batch_size=eval_batch_size)

    probe = torch.nn.Linear(hidden_size, 2).to(device)

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size = batch_size, shuffle=True)

    class_counts = torch.bincount(y)
    class_weights = len(y) / (len(class_counts) * class_counts)

    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.AdamW(probe.parameters(), lr=learning_rate)

    best_f05 = -1
    probe_path = experiment_dir / f"probe_layer_{layer_idx:02}.pt"

    for epoch in range(epochs):
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
        probs, y_dev_eval = get_probs_and_labels(probe, dev_loader, device)

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

            if metrics["f0.5"] > best_f05:
                best_f05 = metrics["f0.5"]
                torch.save(probe.state_dict(), probe_path)

    del X
    del X_dev
    del dev_loader
    del dev_dataset
    del probe

    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()
