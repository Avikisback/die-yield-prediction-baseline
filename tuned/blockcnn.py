"""Does a learned detector beat the derived one on the block readings?

``tuned/blocks.py`` is *derived*: the likelihood ratio for the generator's block
process is written down, the unknown cluster seed is integrated out, and the
result is a bank of circular correlations.  ``tuned/blocksim.py`` bounds it --
ROC-AUC about 0.86 if the seed were known, about 0.74 when it is not.

That is an argument, not a measurement.  This module supplies the measurement:
a 1-D convolutional network over the raw 2,000 readings, trained on the same
dies and scored on the same held-out ones.  A network is exactly the tool that
would find structure a derivation missed -- a second signature, an interaction,
a better nonlinearity -- so if it ties, the derivation is not merely defensible,
it is complete.

The architecture is deliberately shaped like the thing it is competing with,
which is the fair test: circular convolutions (the cluster wraps), a widening
receptive field reaching the cluster's ~200-sample scale, and a pooling layer
that can express ``max``, ``mean`` and ``logsumexp`` over shifts, the last with a
learnable temperature.  Anything the derived statistic computes, this can also
compute; the question is whether it finds more.

This is the one part of the repository a GPU is the right tool for.  The rest of
the pipeline is a diagonal discriminant and a twenty-column penalised logistic --
about six megaflops per optimiser iteration, which is less than the cost of
launching a kernel.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from tuned import blocks, blocksim


class Net:
    """Lazily built so importing this module never requires torch."""

    def __new__(cls, channels: int = 24, dilations=(1, 4, 16, 48)):
        import torch
        from torch import nn

        class Pool(nn.Module):
            """max, mean and a soft-max over shifts, per channel."""

            def __init__(self, width: int):
                super().__init__()
                self.temperature = nn.Parameter(torch.zeros(width))

            def forward(self, x):
                # (1/s) * log mean_j exp(s * x_j): the same soft maximum the
                # derived detector uses to integrate out the cluster seed, with
                # s learned per channel rather than derived.
                scale = torch.exp(self.temperature).clamp(0.05, 20.0)[None, :, None]
                soft = (torch.logsumexp(x * scale, dim=2)
                        - np.log(x.shape[2])) / scale[:, :, 0]
                return torch.cat([x.amax(dim=2), x.mean(dim=2), soft], dim=1)

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                layers = []
                previous = 1
                for dilation in dilations:
                    pad = 7 * dilation
                    layers += [
                        nn.Conv1d(previous, channels, kernel_size=15,
                                  dilation=dilation, padding=pad,
                                  padding_mode="circular"),
                        nn.GELU(),
                    ]
                    previous = channels
                self.body = nn.Sequential(*layers)
                self.pool = Pool(channels)
                self.head = nn.Sequential(
                    nn.Linear(3 * channels, 64), nn.GELU(), nn.Linear(64, 1))

            def forward(self, x):
                return self.head(self.pool(self.body(x[:, None, :]))).squeeze(1)

        return Model()


def train(readings: np.ndarray, label: np.ndarray, groups: np.ndarray,
          folds: int = 4, epochs: int = 14, batch: int = 256,
          learning_rate: float = 2e-3, seed: int = 0, verbose: bool = True):
    """Wafer-grouped out-of-fold scores from the network."""
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"  device: {device}"
              + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda"
                 else ""), flush=True)
    centre = float(np.median(readings))
    scale = float(1.4826 * np.median(np.abs(readings - centre))) or 1.0
    x_all = ((readings - centre) / scale).astype(np.float32)
    y_all = label.astype(np.float32)

    unique = np.unique(groups)
    rng = np.random.default_rng(seed)
    assignment = {g: i % folds for i, g in enumerate(rng.permutation(unique))}
    fold_of = np.array([assignment[g] for g in groups])

    out = np.zeros(len(label), dtype=np.float64)
    positive_weight = float((y_all == 0).sum() / max((y_all == 1).sum(), 1))
    for fold in range(folds):
        torch.manual_seed(seed + fold)
        train_index = np.flatnonzero(fold_of != fold)
        test_index = np.flatnonzero(fold_of == fold)
        model = Net().to(device)
        optimiser = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                                      weight_decay=1e-4)
        schedule = torch.optim.lr_scheduler.OneCycleLR(
            optimiser, max_lr=learning_rate,
            total_steps=epochs * max(1, len(train_index) // batch))
        loss_fn = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(np.sqrt(positive_weight), device=device))

        x_train = torch.from_numpy(x_all[train_index])
        y_train = torch.from_numpy(y_all[train_index])
        for epoch in range(epochs):
            model.train()
            order = torch.randperm(len(train_index))
            total = 0.0
            for start in range(0, len(order) - batch + 1, batch):
                index = order[start:start + batch]
                xb = x_train[index].to(device, non_blocking=True)
                yb = y_train[index].to(device, non_blocking=True)
                optimiser.zero_grad(set_to_none=True)
                loss = loss_fn(model(xb), yb)
                loss.backward()
                optimiser.step()
                schedule.step()
                total += float(loss.detach())
            if verbose and (epoch + 1) % 4 == 0:
                print(f"    fold {fold} epoch {epoch + 1:2d} loss "
                      f"{total / max(1, len(order) // batch):.4f}", flush=True)

        model.eval()
        with torch.no_grad():
            scores = []
            for start in range(0, len(test_index), 512):
                chunk = torch.from_numpy(
                    x_all[test_index[start:start + 512]]).to(device)
                scores.append(model(chunk).float().cpu().numpy())
        out[test_index] = np.concatenate(scores)
        if verbose:
            print(f"    fold {fold}: AUC "
                  f"{roc_auc_score(y_all[test_index], out[test_index]):.4f}",
                  flush=True)
    return out


def derived_scores(readings: np.ndarray, label: np.ndarray, groups: np.ndarray,
                   folds: int = 4, seed: int = 0) -> np.ndarray:
    """The same comparison for ``tuned.blocks``, on the same wafer-grouped folds."""
    noise = blocks.NoiseModel.fit(readings)
    columns, _ = blocks.engineer(readings, noise)
    x = np.column_stack([np.asarray(columns[k], dtype=np.float64)
                         for k in sorted(columns)])
    unique = np.unique(groups)
    rng = np.random.default_rng(seed)
    assignment = {g: i % folds for i, g in enumerate(rng.permutation(unique))}
    fold_of = np.array([assignment[g] for g in groups])
    out = np.zeros(len(label), dtype=np.float64)
    for fold in range(folds):
        train_index = np.flatnonzero(fold_of != fold)
        test_index = np.flatnonzero(fold_of == fold)
        model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                              LogisticRegression(C=0.02, max_iter=3000))
        model.fit(x[train_index], label[train_index])
        out[test_index] = model.predict_proba(x[test_index])[:, 1]
    return out


def simulate(n_wafers: int, per_wafer: int, fail_rate: float, seed: int):
    """Dies laid out in wafers, so the folds can be grouped the way the rest is."""
    rng = np.random.default_rng(seed)
    total = n_wafers * per_wafer
    label = (rng.random(total) < fail_rate).astype(np.int8)
    readings, _ = blocksim.simulate(total, label.astype(bool), rng)
    groups = np.repeat(np.arange(n_wafers), per_wafer)
    return readings, label, groups


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wafers", type=int, default=40)
    parser.add_argument("--per-wafer", type=int, default=1_000)
    parser.add_argument("--fail-rate", type=float, default=0.10)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=14)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path,
                        default=Path("results/tuned_ceiling/block_cnn.json"))
    args = parser.parse_args()

    print(f"simulating {args.wafers * args.per_wafer} dies", flush=True)
    started = time.perf_counter()
    readings, label, groups = simulate(args.wafers, args.per_wafer,
                                       args.fail_rate, args.seed)
    print(f"  {int(label.sum())} failures, {time.perf_counter() - started:.0f}s",
          flush=True)

    print("derived likelihood-ratio bank:", flush=True)
    started = time.perf_counter()
    derived = derived_scores(readings, label, groups, args.folds, args.seed)
    derived_seconds = time.perf_counter() - started

    print("1-D convolutional network:", flush=True)
    started = time.perf_counter()
    learned = train(readings, label, groups, args.folds, args.epochs,
                    seed=args.seed)
    learned_seconds = time.perf_counter() - started

    both = np.column_stack([derived, learned])
    result = {
        "dies": int(len(label)),
        "failures": int(label.sum()),
        "derived_auc": float(roc_auc_score(label, derived)),
        "derived_ap": float(average_precision_score(label, derived)),
        "derived_seconds": round(derived_seconds, 1),
        "cnn_auc": float(roc_auc_score(label, learned)),
        "cnn_ap": float(average_precision_score(label, learned)),
        "cnn_seconds": round(learned_seconds, 1),
        "combined_auc": float(roc_auc_score(
            label, derived_scores_combined(both, label, groups, args.folds,
                                           args.seed))),
    }
    for key, value in result.items():
        print(f"  {key:20s} {value}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {args.output}")


def derived_scores_combined(x: np.ndarray, label: np.ndarray, groups: np.ndarray,
                            folds: int, seed: int) -> np.ndarray:
    """Both scores in one logistic, to see whether they carry different things."""
    unique = np.unique(groups)
    rng = np.random.default_rng(seed)
    assignment = {g: i % folds for i, g in enumerate(rng.permutation(unique))}
    fold_of = np.array([assignment[g] for g in groups])
    out = np.zeros(len(label))
    for fold in range(folds):
        train_index = np.flatnonzero(fold_of != fold)
        test_index = np.flatnonzero(fold_of == fold)
        model = make_pipeline(StandardScaler(),
                              LogisticRegression(C=1.0, max_iter=2000))
        model.fit(x[train_index], label[train_index])
        out[test_index] = model.predict_proba(x[test_index])[:, 1]
    return out


if __name__ == "__main__":
    main()
