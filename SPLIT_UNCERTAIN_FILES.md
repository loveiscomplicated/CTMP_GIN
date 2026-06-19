# Split Uncertain Files

Files listed here were not confidently classified as core implementation for the retrospective CTMP-GIN repository during the split.

## Retained For Manual Review

- `src/analysis/*.ipynb`: exploratory notebooks may include original and forecasted analysis mixed together.
- `src/analysis/*.csv`, `src/analysis/*.png`: small analysis outputs were retained when copied, but should be reviewed before publication.
- `src/explainers/results/**`: tracked explanation outputs are small enough to retain, but they are generated artifacts.
- `ㅁㄴㅇㄹ.md`: scratch note with unclear final role.

## Removed From CTMP-GIN If Forecasted-Specific

Forecasted implementation modules, tests, configs, scripts, reports, and large generated analysis CSVs are removed from this repository. Documentation may keep a short reference that the forecasted extension lives in `Forecasted-CTMP-GIN`.
