from __future__ import annotations

SPLIT_SEED = 42  # D_hpo / D_eval external split, generated once.
HPO_SEED = 42  # D_hpo fold generation, deliberately separate namespace.
EVAL_SEEDS = (1, 2, 3)

PROTOCOL_VERSION = "2026-08-03-v1"
ESTIMATOR_VERSION = "categorical_plugin_v1"
HPO_SUBSAMPLE_RATIO = 0.20
EXTERNAL_HPO_RATIO = 0.15
EVAL_FOLDS = 5
HPO_FOLDS = 3
INNER_VAL_RATIO = 0.10
