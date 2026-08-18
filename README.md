# CTMP-GIN

Original retrospective CTMP-GIN repository for episode-level TEDS-D discharge classification.

This repository uses admission and discharge snapshots plus the observed LOS cross-temporal edge feature. The prospective Forecasted CTMP-GIN extension has been split into the sibling `Forecasted-CTMP-GIN` repository.

## Scope

Included:

- CTMP-GIN retrospective model.
- Admission/discharge graph construction.
- Actual LOS edge embedding.
- MI-based and fully connected edge construction.
- GIN, A3TGCN, GIN-GRU, MLP, and XGBoost baselines.
- Original ablation configs such as no gate, fully connected, and no preprocessing.
- Optuna, k-fold CV, evaluation, and explainer utilities.

Excluded:

- Forecasted discharge or LOS predictors.
- Forecast cache generation/reading.
- Joint-consistent, joint-generative, outcome-aware, risk-head, and drift diagnostics.
- Forecasted downstream wrappers and forecasted experiment launchers.

## Data

Place the raw CSV at:

```bash
src/data/raw/TEDS_Discharge.csv
```

Large/raw data is intentionally not tracked in this split repository.

## Environment

Conda is not required. For local CPU/MPS usage:

```bash
python3 -m pip install -U pip
python3 -m pip install .
```

For development/tests:

```bash
python3 -m pip install -r requirements.txt
```

For VastAI/CUDA, use the helper so PyTorch and PyG wheels are installed from
the matching wheel indexes. It also downloads `src/data/raw/TEDS_Discharge.csv`
from the project Google Drive file ID when the file is missing:

```bash
bash scripts/install_pip.sh
```

For dependency-only setup, run `DOWNLOAD_TEDS_DATA=0 bash scripts/install_pip.sh`.

Quick dependency check:

```bash
python -c "import torch; print(torch.__version__)"
python -c "import torch_geometric; print(torch_geometric.__version__)"
```

`rclone` is a system command, not a Python package. Install it with the OS
package manager when you need Google Drive upload/download, for example
`apt install rclone` on VastAI or `brew install rclone` on macOS.

## Run

Single config run:

```bash
python src/main.py --config configs/ctmp_gin.yaml
```

K-fold CV:

```bash
python src/main.py --config configs/ctmp_gin.yaml --prepare_cv_only
python src/main.py --config configs/ctmp_gin.yaml --fold 0 --cv_run_dir <CV_RUN_DIR>
python src/main.py --config configs/ctmp_gin.yaml --finalize_cv --cv_run_dir <CV_RUN_DIR>
```

Vast.ai CV launcher:

```bash
bash run_vast_cv.sh ctmp_gin configs/ctmp_gin.yaml 1
```

XGBoost:

```bash
python src/main.py --config configs/xgboost.yaml
```

Protocol multi-GPU pipeline, excluding XGBoost:

```bash
export PROTOCOL_OPTUNA_STORAGE='postgresql+psycopg2://USER:PASSWORD@HOST:5432/DB'
export DISCORD_WEBHOOK_URL='https://discord.com/api/webhooks/...'
python scripts/protocol_multigpu_pipeline.py \
  --run-dir new_runs/full_neural_ablation \
  --root src/data \
  --gpus auto \
  --dry-run

python scripts/protocol_multigpu_pipeline.py \
  --run-dir new_runs/full_neural_ablation \
  --root src/data \
  --gpus auto
```

Each training subprocess sees only one GPU through `CUDA_VISIBLE_DEVICES`.
The launcher uses all listed GPUs by scheduling independent HPO/evaluation
jobs across them. Discord and PostgreSQL Optuna storage are required.

## CLI Overrides

```bash
python src/main.py --config configs/ctmp_gin.yaml \
  --device cuda \
  --batch_size 64 \
  --learning_rate 0.0005 \
  --epochs 50 \
  --seed 123 \
  --decision_threshold 0.5 \
  --binary 1
```

## Outputs

Runs write to `runs/`, which is ignored:

- `config.final.yaml`
- `metrics.jsonl`
- `checkpoints/`
- `edge_index.pt`
- k-fold summaries under the CV run directory

## Layout

- `configs/`: retrospective model/baseline/ablation configs.
- `scripts/`: MI/Vast helper scripts.
- `src/data_processing/`: canonical preprocessing, dataset, split, and edge utilities.
- `src/models/`: CTMP-GIN and baseline models.
- `src/trainers/`: single-run, k-fold, and Optuna training code.
- `src/explainers/`: retrospective explanation utilities.
- `tests/`: smoke/unit tests retained for the split.
