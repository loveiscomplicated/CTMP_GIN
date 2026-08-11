#!/usr/bin/env bash
set -euo pipefail

cat >&2 <<'EOF'
run_optuna_vast.sh is deprecated and disabled.

This script used src.trainers.run_parameter_search_optuna, which is the legacy
HPO path. It is not valid for the CTMP-GIN re-experiment protocol.

Use src.protocol.runner stages instead:

  uv run python -m src.protocol.runner --stage prepare ...
  uv run python -m src.protocol.runner --stage edge-pilot ...
  uv run python -m src.protocol.runner --stage hpo ...
  uv run python -m src.protocol.runner --stage top5-reeval ...
  uv run python -m src.protocol.runner --stage evaluate ...

Required protocol arguments include --config, --root, --run-dir, --codebook,
and for HPO/evaluation the saved --graph-config / --selected-config artifacts.
EOF

exit 2
