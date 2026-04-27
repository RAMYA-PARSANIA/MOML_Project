"""
Multi-Objective Optimization for Image Classification
======================================================
Optimizes a CNN on Fashion-MNIST using Optuna's NSGA-II sampler.

Objectives (3):
    1. Maximize classification accuracy
    2. Minimize inference time (ms per batch)
    3. Minimize model size (number of parameters)

Decision Variables:
    - n_conv_layers   : Number of convolutional layers (1–4)
    - channels_i      : Number of channels per conv layer (16–128)
    - learning_rate   : Log-uniform in [1e-4, 1e-1]
    - batch_size      : {32, 64, 128, 256}
    - n_epochs        : Training epochs (3–15)
    - dropout_rate    : Dropout probability (0.0–0.5)
    - optimizer_type  : {Adam, SGD, RMSprop}
    - input_resolution: Image resize {28, 24, 20, 16}

"""

import os
import time
import json
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import optuna
from optuna.samplers import NSGAIISampler

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_TRIALS       = 60          # total number of Optuna trials
SEED           = 42
DATA_DIR       = "./data"
RESULTS_DIR    = "./results"
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------
def get_dataloaders(batch_size: int, input_resolution: int):
    """Return train and test DataLoaders for Fashion-MNIST."""
    tf = transforms.Compose([
        transforms.Resize((input_resolution, input_resolution)),
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),   # Fashion-MNIST stats
    ])
    train_ds = datasets.FashionMNIST(DATA_DIR, train=True,  download=True, transform=tf)
    test_ds  = datasets.FashionMNIST(DATA_DIR, train=False, download=True, transform=tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=True)
    return train_loader, test_loader


# ---------------------------------------------------------------------------
# Dynamic CNN builder
# ---------------------------------------------------------------------------
class DynamicCNN(nn.Module):
    """
    A CNN whose depth and width are controlled by hyperparameters.
    Each conv block: Conv2d → BatchNorm → ReLU → MaxPool2d(2)
    Followed by a flatten → FC → ReLU → Dropout → FC(10)
    """

    def __init__(self, n_conv_layers: int, channels: list[int],
                 dropout_rate: float, input_resolution: int):
        super().__init__()

        conv_blocks = []
        in_ch = 1  # grayscale
        spatial = input_resolution

        for i in range(n_conv_layers):
            out_ch = channels[i]
            conv_blocks.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1))
            conv_blocks.append(nn.BatchNorm2d(out_ch))
            conv_blocks.append(nn.ReLU(inplace=True))
            conv_blocks.append(nn.MaxPool2d(2))
            in_ch = out_ch
            spatial = spatial // 2

        self.features = nn.Sequential(*conv_blocks)
        flat_dim = in_ch * spatial * spatial

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def count_parameters(model: nn.Module) -> int:
    """Return total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Training & Evaluation
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()


@torch.no_grad()
def evaluate(model, loader, device):
    """Return accuracy (fraction) on the given loader."""
    model.eval()
    correct, total = 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        preds = model(X).argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return correct / total


@torch.no_grad()
def measure_inference_time(model, loader, device, n_batches=10):
    """Average inference time (ms) per batch over n_batches."""
    model.eval()
    times = []
    for i, (X, _) in enumerate(loader):
        if i >= n_batches:
            break
        X = X.to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = model(X)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)  # ms
    return float(np.mean(times))


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------
def objective(trial: optuna.Trial):
    """
    Multi-objective function for Optuna.
    Returns: (neg_accuracy, inference_time_ms, n_parameters)
    All three are MINIMIZED by NSGA-II.
    """

    # ---- Sample decision variables ----
    n_conv_layers    = trial.suggest_int("n_conv_layers", 1, 4)
    channels = [
        trial.suggest_categorical(f"channels_{i}",
                                  [16, 32, 48, 64, 96, 128])
        for i in range(n_conv_layers)
    ]
    learning_rate    = trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True)
    batch_size       = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    n_epochs         = trial.suggest_int("n_epochs", 3, 15)
    dropout_rate     = trial.suggest_float("dropout_rate", 0.0, 0.5)
    optimizer_name   = trial.suggest_categorical("optimizer_type",
                                                  ["Adam", "SGD", "RMSprop"])
    input_resolution = trial.suggest_categorical("input_resolution", [16, 20, 24, 28])

    # Guard: spatial dimension must survive all pooling layers
    # After n_conv_layers MaxPool2d(2), spatial = input_resolution / 2^n
    final_spatial = input_resolution // (2 ** n_conv_layers)
    if final_spatial < 1:
        # Prune infeasible configurations
        raise optuna.TrialPruned()

    # ---- Build model ----
    model = DynamicCNN(n_conv_layers, channels, dropout_rate,
                       input_resolution).to(DEVICE)
    n_params = count_parameters(model)

    # ---- Optimizer ----
    opt_map = {
        "Adam":    optim.Adam,
        "SGD":     optim.SGD,
        "RMSprop": optim.RMSprop,
    }
    optimizer = opt_map[optimizer_name](model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # ---- Data ----
    train_loader, test_loader = get_dataloaders(batch_size, input_resolution)

    # ---- Train ----
    for epoch in range(n_epochs):
        train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)

    # ---- Final evaluation ----
    accuracy       = evaluate(model, test_loader, DEVICE)
    inference_time = measure_inference_time(model, test_loader, DEVICE)

    print(f"  Trial {trial.number:>3d} | "
          f"acc={accuracy:.4f}  infer={inference_time:.2f}ms  "
          f"params={n_params:,}  layers={n_conv_layers}  "
          f"res={input_resolution}  opt={optimizer_name}")

    # All objectives are MINIMIZED: negate accuracy
    return -accuracy, inference_time, n_params


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"Device: {DEVICE}")
    print(f"Running {N_TRIALS} NSGA-II trials ...\n")

    # Create multi-objective study
    sampler = NSGAIISampler(seed=SEED)
    study = optuna.create_study(
        directions=["minimize", "minimize", "minimize"],  # -acc, time, params
        sampler=sampler,
        study_name="MOO_FashionMNIST",
    )

    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    # ---- Extract results ----
    records = []
    for t in study.trials:
        if t.state == optuna.trial.TrialState.COMPLETE:
            records.append({
                "trial":           t.number,
                "accuracy":        -t.values[0],       # un-negate
                "inference_ms":    t.values[1],
                "n_parameters":    int(t.values[2]),
                **t.params,
            })

    df_all = pd.DataFrame(records)
    df_all.to_csv(os.path.join(RESULTS_DIR, "all_trials.csv"), index=False)
    print(f"\nSaved {len(df_all)} completed trials → {RESULTS_DIR}/all_trials.csv")

    # ---- Pareto front ----
    pareto_trials = study.best_trials
    pareto_records = []
    for t in pareto_trials:
        pareto_records.append({
            "trial":        t.number,
            "accuracy":     -t.values[0],
            "inference_ms": t.values[1],
            "n_parameters": int(t.values[2]),
            **t.params,
        })

    df_pareto = pd.DataFrame(pareto_records)
    df_pareto.to_csv(os.path.join(RESULTS_DIR, "pareto_front.csv"), index=False)
    print(f"Pareto front: {len(df_pareto)} non-dominated solutions "
          f"→ {RESULTS_DIR}/pareto_front.csv")

    # ---- Save study metadata ----
    meta = {
        "n_trials":            N_TRIALS,
        "n_completed":         len(df_all),
        "n_pareto":            len(df_pareto),
        "device":              str(DEVICE),
        "seed":                SEED,
        "objectives":          ["-accuracy (min)", "inference_ms (min)",
                                "n_parameters (min)"],
        "decision_variables":  ["n_conv_layers", "channels_i", "learning_rate",
                                "batch_size", "n_epochs", "dropout_rate",
                                "optimizer_type", "input_resolution"],
    }
    with open(os.path.join(RESULTS_DIR, "study_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("\nDone! Run analyze_results.py to generate plots and metrics.\n")


if __name__ == "__main__":
    main()
