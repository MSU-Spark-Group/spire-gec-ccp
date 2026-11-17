from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from spire import callbacks as callback_utils
from spire import spire

FIXED = Path(__file__).resolve().parent / "data" / "fixed"
N_FRAMES = 20
GRID_HEIGHT, GRID_WIDTH = 120, 160
BATCH_SIZE = 8
EPOCHS = 900
LEARNING_RATE = 5e-5
MIN_LEARNING_RATE = 1e-5
WARMUP_EPOCHS = 3
SEED = 42


class LinearWarmup(tf.keras.callbacks.Callback):

    def __init__(self, target_lr, warmup_epochs):
        super().__init__()
        self.target_lr = float(target_lr)
        self.warmup_epochs = int(warmup_epochs)

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.warmup_epochs:
            tf.keras.backend.set_value(self.model.optimizer.lr, self.target_lr * (epoch + 1) / self.warmup_epochs)


def load_cases(paths):
    cases = []
    for path in paths:
        with np.load(path) as d:
            x, y = d["x"], d["y"]
        if x.shape != (N_FRAMES, GRID_HEIGHT, GRID_WIDTH, 3) or y.shape != (N_FRAMES, GRID_HEIGHT, GRID_WIDTH, 1):
            raise ValueError(f"{path}: array shapes {x.shape}, {y.shape}")
        cases.append((x, y))
    return cases


def batches(cases, batch_size):
    while True:
        for start in range(0, len(cases), batch_size):
            batch = cases[start:start + batch_size]
            yield np.stack([c[0] for c in batch]), np.stack([c[1] for c in batch])


def main():
    parser = argparse.ArgumentParser(description="Train the SPIRE reconstruction model.")
    parser.add_argument("--data-dir", required=True, help="folder with train_*.npz and val_*.npz")
    parser.add_argument("--out-dir", required=True, help="folder for the run directory")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help=f"number of epochs (default {EPOCHS})")
    parser.add_argument("--causal-encoder", action="store_true", help="causal encoder: each frame sees only earlier frames")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    train_paths, val_paths = sorted(data_dir.glob("train_*.npz")), sorted(data_dir.glob("val_*.npz"))
    if not train_paths or not val_paths:
        raise FileNotFoundError(f"{data_dir} needs train_*.npz and val_*.npz files")

    tf.keras.utils.set_random_seed(SEED)
    loss_weight = np.load(FIXED / "loss_weight.npy").astype(np.float32)
    model = spire.build_spire_reconstruction_model(FIXED / "physics_guided_coefficients.npz", loss_weight,
                                                   causal_encoder=args.causal_encoder, n_frames=N_FRAMES,
                                                   grid_height=GRID_HEIGHT, grid_width=GRID_WIDTH)
    callbacks, run_name = callback_utils.create_callbacks_from_model(model, parent_folder=args.out_dir, timezone="UTC")
    run_dir = Path(args.out_dir) / run_name
    for cb in callbacks:
        if isinstance(cb, tf.keras.callbacks.ReduceLROnPlateau):
            cb.min_lr = MIN_LEARNING_RATE
    tf.keras.backend.set_value(model.optimizer.lr, LEARNING_RATE)
    callbacks = [LinearWarmup(LEARNING_RATE, WARMUP_EPOCHS)] + callbacks

    train_cases, val_cases = load_cases(train_paths), load_cases(val_paths)
    print(f"[train] {len(train_cases)} training and {len(val_cases)} validation cases; run directory {run_dir}")
    history = model.fit(
        batches(train_cases, BATCH_SIZE),
        steps_per_epoch=math.ceil(len(train_cases) / BATCH_SIZE),
        epochs=args.epochs,
        callbacks=callbacks,
        validation_data=batches(val_cases, BATCH_SIZE),
        validation_steps=math.ceil(len(val_cases) / BATCH_SIZE),
        verbose=1,
    )
    pd.DataFrame(history.history).to_csv(run_dir / "training_history.csv", float_format="%.15f", index=False)
    print(f"[train] done; checkpoints in {run_dir}")


if __name__ == "__main__":
    main()
