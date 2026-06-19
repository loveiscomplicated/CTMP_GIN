# Repository Split Report: CTMP-GIN

## Generated Path

`/Users/jeong-yunseong/Documents/programming/CTMP-GIN`

## Final Tree Summary

Top-level contents retained for the retrospective repository:

- `src/`: original CTMP-GIN, GIN, A3TGCN, GIN-GRU, XGBoost, preprocessing, trainers, metrics, explainers.
- `configs/`: original model, baseline, ablation, Optuna, and CV configs.
- `scripts/`: MI and infrastructure helper scripts retained for original experiments.
- `tests/`: retrospective CTMP-GIN LOS encoder/model smoke tests.
- `README.md`, `REPO_SPLIT_PLAN.md`, `SOURCE_REPO_COMMIT.txt`, `SPLIT_UNCERTAIN_FILES.md`.

## Removed Main Files And Directories

- Forecasted model modules: `src/models/discharge_predictor/`, `src/models/forecasted_ctmp_gin/`, `src/models/forecast_inputs.py`.
- Forecasted trainers: forecasted discharge/LOS/pipeline diagnostics, joint-consistent predictor, outcome-aware stage2 runners.
- Forecasted entrypoints: `src/main_discharge.py`, `src/main_los_ordinal.py`.
- Forecasted datasets: discharge and LOS prediction dataset modules.
- Forecasted diagnostics and analysis scripts, including joint drift, LOS breakdown, joint plausibility, and fallback ablation files.
- Forecasted configs, launch scripts, docs, reports, and tests.
- Large generated analysis artifact: `src/analysis/gated_fusion_exports/gated_fusion_samples.csv`.

## Shared Core Retained

- CTMP-GIN model backbone and actual LOS edge embedding.
- Original admission/discharge graph construction and MI edge construction.
- Data preprocessing utilities and canonical split helpers.
- GIN, A3TGCN two-point, GIN-GRU two-point, MLP admission baseline, and XGBoost baseline.
- CV, Optuna, evaluation, metrics, early stopping, and save/load utilities.

## Mixed File Patch Details

- `src/main.py`: removed forecast/joint/outcome-aware CLI paths and retained original single-run/CV dispatch.
- `src/trainers/base.py`: removed forecast metadata, soft-discharge, LOS/discharge provider overrides, and forecast checkpoint extras.
- `src/trainers/run_kfold_cv.py`: rebuilt as original-only k-fold runner with prepare/single-fold/finalize flow.
- `src/models/ctmp_gin/*`: removed forecast distribution input handling, soft discharge, predicted LOS distribution paths, and forecast metadata resolver while retaining hard LOS edge encoding.
- `src/models/gin/model.py`: removed forecast distribution/soft-discharge branches and retained retrospective entity embedding path.
- `src/trainers/run_single_experiment.py`, `src/main_runpod.py`, Optuna runners, and `run_vast_cv.sh`: removed forecast defaults/bootstrap dependencies.

## Unresolved Uncertain Files

See `SPLIT_UNCERTAIN_FILES.md`.

- `src/analysis/*.ipynb`, small `src/analysis/*.csv`, and small `src/analysis/*.png`.
- `src/explainers/results/**` generated explanation outputs.
- `ㅁㄴㅇㄹ.md` scratch note.

## Forecast Keyword Verification

Command:

```bash
rg -n -i "forecast|predicted_D|predicted_LOS|joint_consistent|outcome_aware|joint_generative|risk_head|drift" .
```

Result:

- Implementation files: no matches.
- Documentation or review files only: `README.md`, `REPO_SPLIT_PLAN.md`, `SPLIT_UNCERTAIN_FILES.md`, `ㅁㄴㅇㄹ.md`.

## Large Artifact Verification

Commands:

```bash
find . -type f -size +50M -print
find . -type f -size +10M -print
```

Result: no files reported.

Excluded artifact directories (`runs/`, `checkpoints/`, `artifacts/`, `wandb/`) are absent. Raw/processed data payloads are absent; only `src/data/raw/README_data.md` remains as a data reference note.

## Validation Commands And Results

```bash
python -m compileall src
```

Result: passed.

```bash
pytest -q
```

Result: passed, `6 passed, 1 warning`.

```bash
python -c "import src; print('import ok')"
```

Result: passed, printed `import ok`.

```bash
python src/main.py --help
```

Result: passed, original CTMP-GIN CLI help displayed.

## Failed Tests And Causes

None.

## Follow-Up TODO

- Review uncertain exploratory analysis and explanation output files before publication.
- If desired, replace the scratch note `ㅁㄴㅇㄹ.md` with a formal archived note or remove it in a cleanup commit.
