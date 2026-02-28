# Physics-Informed Learning of Jet Stream Variability and Aircraft-Relevant Momentum Transport

Implementation for:

1. ERA5 data ingestion and preprocessing.
2. A residual CNN model for jet-stream-related wind prediction.
3. A physics-informed model with soft constraints:
   - Geostrophic balance residual
   - Thermal wind balance residual
   - Optional momentum/divergence residual
   - Optional vorticity consistency residual
4. Evaluation, visualization, and a simple aviation fuel-burn sensitivity analysis.

## Project Architecture

```text
jetstream-variablility-piml/
├── configs/
│   └── project.yaml                # Central config template (domain, levels, loss weights)
├── data/
│   ├── raw/                        # ERA5 NetCDF/Zarr files (gitignored)
│   └── processed/                  # Model-ready NPZ datasets (gitignored)
├── notebooks/                      # Optional exploration notebooks
├── outputs/                        # Training curves, maps, reports (gitignored)
├── scripts/
│   ├── download_era5.py            # CDS API ERA5 pressure-level downloader
│   ├── prepare_dataset.py          # NetCDF -> model-ready NPZ preprocessing pipeline
│   └── run_mwe.py                  # Minimal working example (synthetic ERA5-like sample)
├── src/
│   └── jetstream_piml/
│       ├── __init__.py
│       ├── aviation.py             # Corridor winds + fuel sensitivity
│       ├── data.py                 # Dataset + DataLoader utilities
│       ├── eval.py                 # Metrics and evaluation helpers
│       ├── models.py               # Baseline CNN
│       ├── physics.py              # Physics residuals and combined physics loss
│       ├── plotting.py             # Plotting utilities
│       └── train.py                # Baseline + physics-informed training loop
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## Exactly What I Need From You

Collect/provide these inputs before real ERA5 training:

1. ERA5 access:
   - ECMWF CDS account
   - API key in `%USERPROFILE%\\.cdsapirc` (Windows)
2. Variable list (minimum):
   - `u` (zonal wind, m/s)
   - `v` (meridional wind, m/s)
   - `t` (temperature, K)
   - `z` (geopotential, m^2/s^2)
3. Pressure levels:
   - Required: `200`, `250`, `300` hPa
4. Spatial domain:
   - Suggested jet domain: latitude `30N-60N`, longitude full NH mid-lat corridor (for example `160W-20W`)
5. Time range:
   - Minimum practical starter: at least 1-2 years of 6-hourly data
   - Better: 10+ years for robust seasonal generalization
6. Temporal resolution:
   - Preferred: 6-hourly
   - Daily possible, but weaker dynamics
7. Storage preference:
   - Raw: NetCDF
   - Training-ready: NPZ for this starter (easy), Zarr for larger-scale training

## ERA5 Dataset Specification

Recommended initial setup:

1. Dataset: `reanalysis-era5-pressure-levels`
2. Variables: `u`, `v`, `t`, `z`
3. Levels: `200`, `250`, `300` hPa
4. Region: `30N-60N`, broad longitudinal belt relevant to transcontinental/transatlantic routes
5. Resolution: native ERA5 pressure-level grid
6. Time frequency: `00/06/12/18 UTC`

Minimal data volume to train a simple model:

1. Time samples:
   - 6-hourly for 2 years gives ~2920 time steps
   - After next-step pairing, ~2919 supervised samples
2. A quick prototype can run with ~500-1000 samples.

## Physics Terms Implemented

Let `f = 2 * Omega * sin(phi)` be Coriolis parameter.

1. Geostrophic residual:
   - `u_g = -(1/f) * dPhi/dy`
   - `v_g =  (1/f) * dPhi/dx`
   - Penalize mismatch between predicted 250 hPa wind and geostrophic wind from `z250`.
2. Thermal-wind residual (200-300 hPa layer):
   - `du/dlnp = -(R/f) * dT/dy`
   - `dv/dlnp =  (R/f) * dT/dx`
   - Penalize mismatch between predicted vertical shear and temperature-gradient-driven shear.
3. Momentum/divergence residual (optional):
   - Penalize horizontal divergence at jet level.
4. Vorticity consistency residual (optional):
   - `zeta = dv/dx - du/dy`
   - Penalize mismatch between predicted relative vorticity and geostrophic relative vorticity.

Total loss:

`L = L_data + lambda_geo * L_geo + lambda_tw * L_tw + lambda_div * L_div + lambda_vort * L_vort`

Robust training defaults:

1. Data term can use Huber loss (`data_loss: huber`) to reduce outlier sensitivity.
2. Physics coefficients can warm up over early epochs (`physics_warmup_epochs`) to improve optimization stability.
3. Gradient clipping and weight decay are enabled from config.

## Quickstart

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run minimal working example (no ERA5 download needed):

```bash
python scripts/run_mwe.py
```

This will:

1. Build a synthetic ERA5-like dataset.
2. Train baseline model for a few epochs.
3. Train physics-informed model for a few epochs.
4. Print comparison metrics:
   - Jet position error
   - Jet strength error
   - Physics residual magnitudes
5. Generate plots in `outputs/`.

## ERA5 Workflow (Real Data)

1. Download ERA5:

```bash
python scripts/download_era5.py --start 2018-01 --end 2020-12 --output data/raw/era5_2018_2020.nc
```

2. Preprocess into training-ready NPZ:

```bash
python scripts/prepare_dataset.py --input "data/raw/*.nc" --output data/processed/era5_jet_dataset.npz
```

3. Use the same training utilities shown in `scripts/run_mwe.py`, replacing synthetic dataset path with the ERA5 NPZ.

## Aviation-Relevant Analysis Included

Implemented in `src/jetstream_piml/aviation.py`:

1. Predefined corridor: JFK -> LAX
2. Extract predicted winds along route
3. Compute along-track and headwind/tailwind components
4. Estimate fuel-burn sensitivity using a simple linear model:
   - `% fuel change = beta * mean_headwind (m/s)`

This is intentionally simple and interpretable for first-pass analysis.

## Faster Repeated Training (Cached Normalization)

If startup is slow, cache input normalization once and reuse it:

```bash
python scripts/cache_input_norm.py --output data/processed/input_norm_stats.npz
```

Then run training with cached stats (skips expensive normalization recompute):

```bash
python training/train_entry.py --input-norm data/processed/input_norm_stats.npz --epochs 20 --device cpu
```

Optional quick subset controls for debugging:

```bash
python training/train_entry.py --quick
python training/train_entry.py --time-stride 8 --max-samples 512 --input-norm data/processed/input_norm_stats.npz
```
