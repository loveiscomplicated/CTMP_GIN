from __future__ import annotations

import argparse
import sys
from textwrap import dedent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deprecated legacy Optuna entrypoint. Use src.protocol.runner instead.",
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--init-only", action="store_true")
    parser.add_argument("--study-name", type=str, default=None)
    parser.add_argument("--n-trials", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--run-dir", type=str, default="runs/protocol")
    parser.add_argument("--graph-config", type=str, default=None)
    parser.add_argument("--codebook", type=str, default=None)
    parser.add_argument("--storage", type=str, default=None)
    return parser.parse_args()


def _protocol_hint(args: argparse.Namespace) -> str:
    config = args.config or "<config.yaml>"
    run_dir = args.run_dir or "<run-dir>"
    graph_config = args.graph_config or f"{run_dir}/graph_config.json"
    codebook = args.codebook or "<teds-d-codebook.json>"
    study_name = args.study_name or "<model>_protocol"
    n_trials = args.n_trials or 100
    storage = args.storage or "$PROTOCOL_OPTUNA_STORAGE"
    return dedent(
        f"""
        This legacy Optuna entrypoint has been disabled.

        It used the pre-protocol HPO path and is not valid for the CTMP-GIN
        re-experiment protocol. In particular, it did not enforce D_hpo/D_eval
        separation, graph-pilot artifacts, codebook vocabularies, or the fixed
        comparison/ablation policy.

        Use the protocol runner instead, for example:

          uv run python -m src.protocol.runner --stage prepare \\
            --config {config} --root src/data --run-dir {run_dir} --codebook {codebook}

          uv run python -m src.protocol.runner --stage edge-pilot \\
            --config {config} --root src/data --run-dir {run_dir} --codebook {codebook}

          uv run python -m src.protocol.runner --stage hpo \\
            --config {config} --root src/data --run-dir {run_dir} \\
            --codebook {codebook} --graph-config {graph_config} \\
            --study-name {study_name} --n-trials {n_trials} \\
            --storage {storage}
        """
    ).strip()


def run_optuna(*args, **kwargs):
    raise RuntimeError(_protocol_hint(argparse.Namespace(
        config=kwargs.get("config_path"),
        run_dir="runs/protocol",
        graph_config=None,
        codebook=None,
        study_name=kwargs.get("study_name"),
        n_trials=kwargs.get("n_trials"),
    )))


def main() -> None:
    print(_protocol_hint(parse_args()), file=sys.stderr)
    raise SystemExit(2)


if __name__ == "__main__":
    main()
