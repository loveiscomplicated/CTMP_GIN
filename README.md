## Project Overview

Temporal graph-based modeling for TEDS-D discharge data. Training runs are driven by YAML configs and saved under `runs/`.

## Data

Place the raw CSV in `src/data/raw/TEDS_Discharge.csv`. The preprocessing step will generate
`src/data/raw/missing_corrected.csv` on first run and reuse it afterwards.

## Environment Setup (Conda)

Tested with **Python 3.12** and a **Conda** environment.

### 1) Create the environment from `environment.yml`

```bash
conda env create -f environment.yml
conda activate pyg_2
```

### 2) (Optional) Install pip packages

If PyG extensions are not fully resolved by conda, install the remaining dependencies via pip:

```bash
pip install -r requirements.txt
```

### 3) Verify installation

```bash
python -c "import torch; print(torch.__version__)"
python -c "import torch_geometric; print(torch_geometric.__version__)"
```

## Train

```bash
python src/main.py --config configs/ctmp_gin.yaml
```

### Config Overrides (CLI)

You can override key settings from the config (example):

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

Each run creates a folder under `runs/` containing:
- `config.final.yaml`: resolved config used for the run
- `metrics.jsonl`: per-epoch metrics
- `checkpoints/`: `last.pt`, `best.pt`, and periodic checkpoints

## Repo Layout

- `src/main.py`: entry point for training
- `configs/`: training configs (only `ctmp_gin.yaml` is populated)
- `src/data/`: raw and processed data cache
- `src/data_processing/`: preprocessing, dataset, and edge construction
- `src/models/`: model definitions
- `src/trainers/`: training and evaluation loops
- `src/utils/`: experiment logging and helpers
