# SPIRE GEC CCP

TensorFlow code of **SPIRE** (Spatiotemporal Physics-guided Inference and REconstruction), a spatiotemporal encoder–decoder network for the electron density ($n_e$) of an argon capacitively coupled plasma (CCP) in the Gaseous Electronics Conference (GEC) reference cell.
The model in this repository is the reconstruction model: from the electron density at three probe windows over one RF period, it reconstructs the time-resolved two-dimensional $n_e$ field.
Its encoder opens with **physics-guided upsampling**, which takes scaling laws of the sheath as priors, and continues with Conv3D blocks; a ConvLSTM2D + Conv3DTranspose decoder follows.

This repository holds the model, the training entry point, and the script that reconstructs one case.
The simulation dataset of the paper is archived on Zenodo: [10.5281/zenodo.21708514](https://doi.org/10.5281/zenodo.21708514).

## Repository Layout

```text
spire-gec-ccp/
├── spire/
│   ├── spire.py          # the model: encoder (physics-guided upsampling layer, Conv3D blocks), ConvLSTM2D decoder
│   └── callbacks.py      # checkpointing, learning-rate reduction, early stopping
├── train.py              # training entry point
├── predict.py            # reconstruction of one case: its error and a picture
├── requirements.txt      # Python packages (TensorFlow 2.17.1, Keras 2 API)
├── slurm/                # SLURM job templates (edit the marked placeholders)
│   ├── train_template.sbatch     # training on one GPU
│   └── predict_template.sbatch   # reconstruction of one case
├── tests/
│   └── test_spire.py     # unit tests of the model parts; they need no data
├── Dockerfile            # environment image
├── CITATION.cff
└── LICENSE               # MIT
```

## Main Code

### `spire/spire.py`

The reconstruction model, built by `build_spire_reconstruction_model(coefficient_file, loss_weight, causal_encoder=False, n_frames=20, grid_height=120, grid_width=160)`.
Its stages, in order:

- **Input** `probes` — `(batch, 20, 120, 160, 3)`: the probe input on the model grid at 20 sampled times of one RF period.
- **`encoder(probes, coefficient_file, causal)`** — the encoder: the physics-guided upsampling layer, then the Conv3D stem and blocks; it returns the encoded features and the physics-guided field.
  - **`PhysicsGuidedUpsampling`** — Keras layer, the first stage of the encoder.
    - `operating_condition(probes)` — estimates the operating condition from the probe input itself.
    - `physics_guided_field(p)` — evaluates the scaling laws of the sheath at that operating condition: the electron edge, the sheath-aligned coordinate, and the $n_e$ law at each value of that coordinate.
    - `call(probes)` — returns the physics-guided $n_e$ field, `(batch, 20, 120, 160, 1)`.
  - **Conv3D stem and four blocks** of 1, 3, 3 and 4 units, with 8, 8, 32, 56 and 56 output channels (`stem`, `block`, `unit`).
    Each unit is an expansion convolution, a depthwise convolution, a squeeze-and-excitation gate and a projection, with a shortcut around them.
    `causal=True` builds the causal form: every Conv3D pads on the past side only, and the squeeze-and-excitation averages over the frames up to the current one.
- **`decoder(features, physics_guided_field, ...)`** — ConvLSTM2D (64 channels), four Conv3DTranspose layers (32, 16, 16, 8 channels) back to the model grid, the physics-guided field joined to their output, and four ConvLSTM2D layers (64, 32, 16, 1 channels).
- **Output** — `(batch, 20, 120, 160, 1)`: the standardized $\log_{10} n_e$ field, zero outside the discharge.

Loss and metric:

- `masked_mse(loss_weight)` — mean squared error over the cells inside the discharge.
- `MaskedRMSE` — root-mean-square error over the same cells.

The model is compiled with Adam and gradient clipping, and has about one million parameters.
Checkpoints save **weights only** (`.h5`).

### `train.py`

Training entry point (`python train.py --help` for the arguments):

- `--data-dir` — folder with the `train_*.npz` and `val_*.npz` case files
- `--out-dir` — folder in which the run directory is created
- `--epochs` — number of epochs (the default is the full training setting)
- `--causal-encoder` — train the causal form of the encoder

The training setting is the set of constants at the top of the file: batch size, number of epochs, learning rate with its warm-up and its floor, and the random seed.
A run writes its checkpoints (`model-<epoch>-<loss>-<rmse>.h5`, kept at the lowest validation loss) and `training_history.csv` to `<out-dir>/spire_reconstruction_<time>/`.

### `predict.py`

Reconstruction of one case with a checkpoint written by `train.py`:

- `--checkpoint` — the `.h5` checkpoint
- `--case` — one case file (`.npz`)
- `--frame` — which of the 20 sampled times to draw
- `--out` — the picture file
- `--causal-encoder` — give it when the checkpoint was trained with it

It prints the RMSE of $\log_{10} n_e$ over the discharge and the 20 sampled times, and writes a picture with three panels: the reconstructed field, the fluid-model field, and their difference.

### `spire/callbacks.py`

`create_callbacks_from_model(model, parent_folder, timezone)` creates the time-stamped run directory and returns the training callbacks:

- `ModelCheckpoint` — weights of the epoch with the lowest validation loss
- `ReduceLROnPlateau` — halves the learning rate when the validation loss stops improving
- `EarlyStopping` — stops the training when the validation loss no longer improves, and restores the best weights

### `tests/test_spire.py`

Unit tests of the parts of the model that need no data and no coefficient file:

- the causal form of the convolution and of a unit: the output at a time does not change when later times of the input change
- the output sizes of the stem and of a unit
- `masked_mse` and `MaskedRMSE`: the average runs over the cells inside the discharge only
- the decoder output is zero outside the discharge
- `create_callbacks_from_model` creates the run directory and returns the three callbacks

```bash
python -m unittest discover -s tests
```

## Data

### Case files

`train.py` and `predict.py` read cases in the form the model takes, one `.npz` file per case:

- `x` — `(20, 120, 160, 3)`: the probe input
- `y` — `(20, 120, 160, 1)`: the fluid-model $n_e$ field, standardized $\log_{10} n_e$
- `frame_idx` — `(20,)`: the indices of the 20 sampled times within the RF period

The complete simulation dataset is on Zenodo (CC-BY-4.0): [10.5281/zenodo.21708514](https://doi.org/10.5281/zenodo.21708514).

## Environment

Python 3.11 with the packages of `requirements.txt`: TensorFlow 2.17.1 with the Keras 2 API through `tf-keras`.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export TF_USE_LEGACY_KERAS=1
```

## Citation

The paper is a manuscript under review; its reference will be added here when it is published.
Citation metadata is provided in `CITATION.cff`.

## License

The source code is distributed under the MIT License in `LICENSE`.
