from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from spire import spire

FIXED = Path(__file__).resolve().parent / "data" / "fixed"
N_FRAMES = 20
GRID_HEIGHT, GRID_WIDTH = 120, 160


def main():
    parser = argparse.ArgumentParser(description="Reconstruct one case with a trained SPIRE checkpoint and draw it.")
    parser.add_argument("--checkpoint", required=True, help="model-*.h5 written by train.py")
    parser.add_argument("--case", required=True, help="one case file (.npz)")
    parser.add_argument("--frame", type=int, default=0, help=f"sampled frame 0..{N_FRAMES - 1} to draw (default 0)")
    parser.add_argument("--out", default="spire_prediction.png", help="picture file (default spire_prediction.png)")
    parser.add_argument("--causal-encoder", action="store_true", help="give it when train.py was run with it")
    args = parser.parse_args()
    if not 0 <= args.frame < N_FRAMES:
        raise ValueError(f"--frame must lie in 0..{N_FRAMES - 1}, got {args.frame}")

    const = json.loads((FIXED / "run_constants.json").read_text())
    loss_weight = np.load(FIXED / "loss_weight.npy").astype(np.float32)
    chamber = loss_weight > 0
    with np.load(args.case) as d:
        x, y, frame_idx = d["x"], d["y"], d["frame_idx"]

    model = spire.build_spire_reconstruction_model(FIXED / "physics_guided_coefficients.npz", loss_weight,
                                                   causal_encoder=args.causal_encoder, n_frames=N_FRAMES,
                                                   grid_height=GRID_HEIGHT, grid_width=GRID_WIDTH)
    model.load_weights(args.checkpoint)
    out_z = model.predict(x[None], verbose=0)[0]

    mu, sigma = const["target_mu"], const["target_sigma"]
    pred = np.where(chamber[None, :, :, None], out_z * sigma + mu, np.nan)[..., 0]
    fluid = np.where(chamber[None, :, :, None], y * sigma + mu, np.nan)[..., 0]
    rmse = float(np.sqrt(np.nanmean((pred - fluid) ** 2)))
    print(f"[predict] {Path(args.case).name}: RMSE over the chamber and the {N_FRAMES} frames: {rmse:.6f} (log10 n_e)")

    win = const["physical_window_cm"]
    extent = [win["r"][0], win["r"][1], win["z"][0], win["z"][1]]
    k = args.frame
    lo, hi = np.nanmin([pred[k], fluid[k]]), np.nanmax([pred[k], fluid[k]])
    diff = pred[k] - fluid[k]
    dmax = float(np.nanmax(np.abs(diff)))
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    panels = [(pred[k], "SPIRE reconstruction", "viridis", lo, hi, r"$\log_{10} n_e$ [m$^{-3}$]"),
              (fluid[k], "fluid model", "viridis", lo, hi, r"$\log_{10} n_e$ [m$^{-3}$]"),
              (diff, "reconstruction $-$ fluid model", "RdBu_r", -dmax, dmax, r"$\Delta \log_{10} n_e$")]
    for ax, (field, title, cmap, vmin, vmax, label) in zip(axes, panels):
        im = ax.imshow(field, extent=extent, origin="upper", cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
        ax.set_title(title)
        ax.set_xlabel("$r$ [cm]")
        ax.set_ylabel("$z$ [cm]")
        fig.colorbar(im, ax=ax, label=label, shrink=0.85)
    t = frame_idx[k] / const["frames_per_rf_period"]
    fig.suptitle(f"{Path(args.case).stem}, $t = {t:.2f}\\,T_{{\\mathrm{{RF}}}}$")
    fig.savefig(args.out, dpi=150)
    print(f"[predict] picture written: {args.out}")


if __name__ == "__main__":
    main()
