#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.protocol.ablations import VARIANTS
from src.protocol.constants import EVAL_FOLDS, EVAL_SEEDS
from src.protocol.runner import _namespaced_study_name

try:
    import optuna
except ImportError:
    optuna = None  # type: ignore[assignment]


MAIN_MODELS = {
    "ctmp_gin": "configs/ctmp_gin.yaml",
    "gin": "configs/gin.yaml",
    "a3tgcn_2_points": "configs/a3tgcn_2_points.yaml",
    "gin_gru_2_points": "configs/gin_gru_2_points.yaml",
}

DEFAULT_ABLATION_VARIANTS = (
    "A1",
    "A2",
    "A3",
    "A4",
    "B1",
    "w/o_merged_stream",
    "w/o_gated_fusion",
    "w/o_mi_edge",
    "w/o_preprocessing",
)


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "unnamed"


def parse_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def storage_backend(storage: str) -> str:
    return urlsplit(storage).scheme.split("+", 1)[0].lower()


def require_postgresql_storage(storage: str | None) -> str:
    resolved = storage or os.environ.get("PROTOCOL_OPTUNA_STORAGE") or os.environ.get("OPTUNA_STORAGE")
    if not resolved:
        raise SystemExit("PROTOCOL_OPTUNA_STORAGE or --storage is required")
    if storage_backend(resolved) not in {"postgres", "postgresql"}:
        raise SystemExit(f"PostgreSQL Optuna storage is required, got: {resolved}")
    return resolved


def detect_gpus(value: str) -> list[str]:
    if value != "auto":
        gpus = parse_csv(value)
        if not gpus:
            raise SystemExit("--gpus did not contain any GPU ids")
        return gpus

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible and visible.lower() not in {"all", "none", "-1"}:
        gpus = parse_csv(visible)
        if gpus:
            return gpus

    try:
        output = subprocess.check_output(["nvidia-smi", "-L"], text=True, stderr=subprocess.DEVNULL)
    except (FileNotFoundError, subprocess.CalledProcessError):
        raise SystemExit("Could not auto-detect GPUs. Pass --gpus 0,1,... explicitly.") from None
    gpus = []
    for line in output.splitlines():
        match = re.match(r"GPU\s+(\d+):", line)
        if match:
            gpus.append(match.group(1))
    if not gpus:
        raise SystemExit("nvidia-smi returned no GPUs")
    return gpus


def split_signature(path: Path) -> list[tuple[str, tuple[int, ...], tuple[int, ...], tuple[int, ...]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    signature = []
    for split in payload.get("splits", []):
        signature.append((
            str(split["split_id"]),
            tuple(int(value) for value in split["train_idx"]),
            tuple(int(value) for value in split["val_idx"]),
            tuple(int(value) for value in split["test_idx"]),
        ))
    return signature


def read_eval_split_ids(run_dir: Path) -> list[str]:
    payload = json.loads((run_dir / "d_eval_split_artifact.json").read_text(encoding="utf-8"))
    return [str(split["split_id"]) for split in payload["splits"]]


def expected_eval_split_ids() -> list[str]:
    return [f"seed{seed}_fold{fold}" for seed in EVAL_SEEDS for fold in range(EVAL_FOLDS)]


def log_tail(path: Path, max_lines: int = 80) -> str:
    if not path.exists():
        return f"log file does not exist: {path}"
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    tail = "\n".join(lines[-max_lines:])
    return tail or "(log file is empty)"


def existing_log_counter(log_dir: Path) -> int:
    if not log_dir.exists():
        return 0
    max_counter = 0
    for path in log_dir.glob("*.log"):
        match = re.match(r"(\d+)_", path.name)
        if match:
            max_counter = max(max_counter, int(match.group(1)))
    return max_counter


def json_artifact_complete(path: Path) -> bool:
    if not path.exists() or path.stat().st_size <= 0:
        return False
    if path.suffix != ".json":
        return True
    try:
        json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return True


@dataclass(frozen=True)
class RunContext:
    key: str
    label: str
    config: Path
    run_dir: Path
    variant: str = "full"

    @property
    def is_ablation(self) -> bool:
        return self.variant != "full"

    @property
    def hpo_required(self) -> bool:
        return bool(VARIANTS.get(self.variant, {}).get("hpo", True))

    @property
    def selected_config(self) -> Path:
        if self.variant == "full":
            return self.run_dir / "selected_config.json"
        if self.hpo_required:
            return self.run_dir / f"selected_config_{safe_name(self.variant)}.json"
        raise ValueError(f"{self.variant} inherits the full CTMP-GIN selected config")

    @property
    def evaluation_summary(self) -> Path:
        return self.run_dir / "evaluation_summary.json"


@dataclass(frozen=True)
class Job:
    name: str
    cmd: list[str]
    gpu_required: bool = True


class DiscordNotifier:
    def __init__(self, webhook_url: str | None, bot_name: str, *, dry_run: bool) -> None:
        self.webhook_url = webhook_url
        self.bot_name = bot_name
        self.dry_run = dry_run

    def require_ready(self) -> None:
        if not self.webhook_url:
            raise SystemExit("DISCORD_WEBHOOK_URL is required for the multi-GPU pipeline")
        if self.dry_run:
            print("[DRY_RUN] Discord webhook is set; skipping test message")
            return
        self.send("[PIPELINE_READY] Discord notification check passed")

    def send(self, message: str) -> None:
        print(message, flush=True)
        if self.dry_run:
            return
        if not self.webhook_url:
            raise RuntimeError("DISCORD_WEBHOOK_URL is required")
        response = requests.post(
            self.webhook_url,
            json={"content": message, "username": self.bot_name},
            timeout=15,
        )
        if response.status_code != 204:
            raise RuntimeError(
                f"Discord notification failed: status={response.status_code} body={response.text}"
            )


class ProtocolPipeline:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.repo_root = REPO_ROOT
        self.run_dir = Path(args.run_dir)
        self.root = Path(args.root)
        self.python_bin = args.python_bin
        self.storage = require_postgresql_storage(args.storage)
        self.gpus = detect_gpus(args.gpus)
        self.resume = not args.no_resume
        self.dry_run = bool(args.dry_run)
        self.log_dir = self.run_dir / "launcher_logs"
        self.notifier = DiscordNotifier(
            os.environ.get("DISCORD_WEBHOOK_URL"),
            args.discord_bot_name,
            dry_run=self.dry_run,
        )
        self.job_counter = existing_log_counter(self.log_dir) if self.resume else 0
        self._active: list[tuple[str, Job, subprocess.Popen]] = []

    def validate(self) -> None:
        self.notifier.require_ready()
        raw_dir = self.root / "raw"
        raw_csv = raw_dir / "TEDS_Discharge.csv"
        if not raw_dir.is_dir():
            raise SystemExit(f"ROOT/raw does not exist: {raw_dir}")
        if not raw_csv.is_file():
            raise SystemExit(f"raw data file does not exist: {raw_csv}")
        for config in MAIN_MODELS.values():
            if not Path(config).is_file():
                raise SystemExit(f"missing config: {config}")
        if not self.gpus:
            raise SystemExit("at least one GPU id is required")
        for variant in self.ablation_variants():
            if variant not in VARIANTS:
                raise SystemExit(f"unknown ablation variant: {variant}")
            if VARIANTS[variant].get("source") is not None:
                raise SystemExit(f"{variant} is not a CTMP-GIN ablation")

    def main_contexts(self) -> list[RunContext]:
        return [
            RunContext(
                key=model,
                label=model,
                config=Path(config),
                run_dir=self.run_dir / "models" / model,
            )
            for model, config in MAIN_MODELS.items()
        ]

    def ctmp_context(self) -> RunContext:
        return self.main_contexts()[0]

    def ablation_variants(self) -> list[str]:
        return parse_csv(self.args.ablation_variants) or list(DEFAULT_ABLATION_VARIANTS)

    def ablation_contexts(self) -> list[RunContext]:
        return [
            RunContext(
                key=safe_name(variant),
                label=f"ablation:{variant}",
                config=Path(MAIN_MODELS["ctmp_gin"]),
                run_dir=self.run_dir / "ablations" / safe_name(variant),
                variant=variant,
            )
            for variant in self.ablation_variants()
        ]

    def runner_cmd(
        self,
        stage: str,
        ctx: RunContext,
        *extra: str | Path,
        include_root: bool = True,
    ) -> list[str]:
        cmd = [
            self.python_bin,
            "-m",
            "src.protocol.runner",
            "--stage",
            stage,
            "--config",
            str(ctx.config),
            "--run-dir",
            str(ctx.run_dir),
        ]
        if include_root:
            cmd.extend(["--root", str(self.root)])
        if ctx.variant != "full":
            cmd.extend(["--variant", ctx.variant])
        if self.args.codebook:
            cmd.extend(["--codebook", self.args.codebook])
        cmd.extend(str(item) for item in extra)
        return cmd

    def optuna_args(self, target_completed: int, max_attempted: int, n_trials: int = 1) -> list[str]:
        args = [
            "--storage",
            self.storage,
            "--n-trials",
            str(n_trials),
            "--target-completed-trials",
            str(target_completed),
            "--max-total-trials",
            str(max_attempted),
            "--discord-bot-name",
            self.args.discord_bot_name,
        ]
        if self.args.notify_trials:
            args.append("--notify-trials")
        return args

    def job_env(self, gpu: str | None) -> dict[str, str]:
        env = os.environ.copy()
        env["PROTOCOL_OPTUNA_STORAGE"] = self.storage
        env["PYTHONUNBUFFERED"] = "1"
        env.setdefault("OMP_NUM_THREADS", str(self.args.omp_threads))
        env.setdefault("MKL_NUM_THREADS", str(self.args.omp_threads))
        if gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        return env

    def run_command(self, name: str, cmd: list[str], *, gpu: str | None = None) -> None:
        self.job_counter += 1
        log_path = self.log_dir / f"{self.job_counter:04d}_{safe_name(name)}.log"
        if self.dry_run:
            gpu_text = f" gpu={gpu}" if gpu is not None else ""
            print(f"[DRY_RUN]{gpu_text} {' '.join(cmd)}")
            return
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.notifier.send(f"[JOB_START] {name} gpu={gpu if gpu is not None else 'cpu'} log={log_path}")
        with log_path.open("w", encoding="utf-8") as handle:
            proc = subprocess.Popen(
                cmd,
                cwd=self.repo_root,
                env=self.job_env(gpu),
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            rc = proc.wait()
        if rc != 0:
            self.notifier.send(f"[JOB_FAIL] {name} rc={rc} log={log_path}")
            tail = log_tail(log_path)
            print(f"----- tail {log_path} -----\n{tail}\n----- end tail -----", flush=True)
            self.notifier.send(f"[JOB_FAIL_TAIL] {name}\n```text\n{tail[-1800:]}\n```")
            raise SystemExit(rc)
        self.notifier.send(f"[JOB_DONE] {name} log={log_path}")

    def start_parallel_job(self, job: Job, gpu: str) -> subprocess.Popen:
        self.job_counter += 1
        log_path = self.log_dir / f"{self.job_counter:04d}_{safe_name(job.name)}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.notifier.send(f"[JOB_START] {job.name} gpu={gpu} log={log_path}")
        handle = log_path.open("w", encoding="utf-8")
        proc = subprocess.Popen(
            job.cmd,
            cwd=self.repo_root,
            env=self.job_env(gpu),
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        proc._protocol_log_handle = handle  # type: ignore[attr-defined]
        proc._protocol_log_path = log_path  # type: ignore[attr-defined]
        return proc

    def terminate_active(self) -> None:
        for _, _, proc in self._active:
            if proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        deadline = time.time() + 30
        for _, _, proc in self._active:
            while proc.poll() is None and time.time() < deadline:
                time.sleep(0.2)
            if proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

    def close_proc_log(self, proc: subprocess.Popen) -> None:
        handle = getattr(proc, "_protocol_log_handle", None)
        if handle is not None:
            handle.close()

    def run_parallel(self, stage_name: str, jobs: list[Job]) -> None:
        if not jobs:
            self.notifier.send(f"[STAGE_SKIP] {stage_name}: no jobs")
            return
        if self.dry_run:
            self.notifier.send(f"[STAGE_DRY_RUN] {stage_name}: {len(jobs)} jobs")
            for index, job in enumerate(jobs):
                gpu = self.gpus[index % len(self.gpus)] if job.gpu_required else None
                print(f"[DRY_RUN] gpu={gpu} {' '.join(job.cmd)}")
            return

        pending = deque(jobs)
        free_gpus = deque(self.gpus)
        completed = 0
        self._active = []
        self.notifier.send(f"[STAGE_START] {stage_name}: jobs={len(jobs)} gpus={','.join(self.gpus)}")
        try:
            while pending or self._active:
                while pending and free_gpus:
                    gpu = free_gpus.popleft()
                    job = pending.popleft()
                    proc = self.start_parallel_job(job, gpu)
                    self._active.append((gpu, job, proc))

                next_active = []
                for gpu, job, proc in self._active:
                    rc = proc.poll()
                    if rc is None:
                        next_active.append((gpu, job, proc))
                        continue
                    self.close_proc_log(proc)
                    log_path = getattr(proc, "_protocol_log_path", "unknown")
                    if rc != 0:
                        self.notifier.send(f"[JOB_FAIL] {job.name} gpu={gpu} rc={rc} log={log_path}")
                        if isinstance(log_path, Path):
                            tail = log_tail(log_path)
                            print(f"----- tail {log_path} -----\n{tail}\n----- end tail -----", flush=True)
                            self.notifier.send(f"[JOB_FAIL_TAIL] {job.name}\n```text\n{tail[-1800:]}\n```")
                        self.terminate_active()
                        raise SystemExit(rc)
                    completed += 1
                    free_gpus.append(gpu)
                    remaining = len(pending) + len(next_active)
                    self.notifier.send(
                        f"[JOB_DONE] {job.name} gpu={gpu} completed={completed}/{len(jobs)} remaining={remaining} log={log_path}"
                    )
                self._active = next_active
                if pending or self._active:
                    time.sleep(self.args.poll_seconds)
        except KeyboardInterrupt:
            self.terminate_active()
            raise
        finally:
            for _, _, proc in self._active:
                self.close_proc_log(proc)
            self._active = []
        self.notifier.send(f"[STAGE_DONE] {stage_name}: completed={completed}/{len(jobs)}")

    def artifact_done(self, path: Path) -> bool:
        if not self.resume:
            return False
        complete = json_artifact_complete(path)
        if path.exists() and not complete:
            self.notifier.send(f"[RESUME_REDO] incomplete artifact will be rebuilt: {path}")
        return complete

    def prepare_and_preflight(self, contexts: list[RunContext]) -> None:
        self.notifier.send(f"[STAGE_START] prepare/preflight contexts={len(contexts)}")
        for ctx in contexts:
            prepare_done = (
                self.artifact_done(ctx.run_dir / "d_eval_split_artifact.json")
                and self.artifact_done(ctx.run_dir / "d_hpo_split_artifact.json")
            )
            if not prepare_done:
                self.run_command(
                    f"prepare {ctx.label}",
                    self.runner_cmd("prepare", ctx),
                )
            if not self.artifact_done(ctx.run_dir / "preflight_report.json"):
                self.run_command(
                    f"preflight {ctx.label}",
                    self.runner_cmd("preflight", ctx),
                )
        self.validate_split_consistency(contexts)
        self.notifier.send("[STAGE_DONE] prepare/preflight")

    def validate_split_consistency(self, contexts: list[RunContext]) -> None:
        if self.dry_run:
            self.notifier.send("[DRY_RUN] split consistency check skipped")
            return
        reference = split_signature(self.ctmp_context().run_dir / "d_eval_split_artifact.json")
        for ctx in contexts:
            current = split_signature(ctx.run_dir / "d_eval_split_artifact.json")
            if current != reference:
                raise SystemExit(f"split artifact mismatch: {ctx.label}")

    def hpo_status(self, ctx: RunContext) -> dict[str, int]:
        if optuna is None:
            raise RuntimeError("Optuna is required for HPO status checks")
        requested = f"{ctx.config.stem}_{ctx.variant}" if ctx.is_ablation else f"{ctx.key}_protocol"
        if ctx.is_ablation:
            requested = f"ctmp_gin_{ctx.variant}"
        study_name = _namespaced_study_name(ctx.run_dir, requested)
        try:
            study = optuna.load_study(study_name=study_name, storage=self.storage)
        except (KeyError, ValueError):
            return {"complete": 0, "running": 0, "pruned": 0, "failed": 0, "waiting": 0, "attempted": 0, "total": 0}
        states = optuna.trial.TrialState
        trials = study.get_trials(deepcopy=False)
        counts = {
            "complete": sum(trial.state == states.COMPLETE for trial in trials),
            "running": sum(trial.state == states.RUNNING for trial in trials),
            "pruned": sum(trial.state == states.PRUNED for trial in trials),
            "failed": sum(trial.state == states.FAIL for trial in trials),
            "waiting": sum(trial.state == states.WAITING for trial in trials),
        }
        counts["attempted"] = counts["complete"] + counts["running"] + counts["pruned"] + counts["failed"]
        counts["total"] = len(trials)
        return counts

    def hpo_stage(self, contexts: list[RunContext], *, ablation: bool) -> None:
        target_completed = self.args.ablation_hpo_trials if ablation else self.args.hpo_trials
        max_attempted = self.args.ablation_max_hpo_attempts if ablation else self.args.max_hpo_attempts
        pending_contexts = [ctx for ctx in contexts if not self.artifact_done(ctx.selected_config)]
        if not pending_contexts:
            self.notifier.send("[STAGE_SKIP] ablation-HPO/top5 already complete" if ablation else "[STAGE_SKIP] main HPO/top5 already complete")
            return
        if self.dry_run:
            jobs = []
            for index, ctx in enumerate(pending_contexts):
                stage = "ablation-hpo" if ctx.is_ablation else "hpo"
                jobs.append(Job(
                    name=f"{ctx.label} {stage} trial_job 1/{target_completed}",
                    cmd=self.runner_cmd(
                        stage,
                        ctx,
                        *self.optuna_args(target_completed, max_attempted, n_trials=1),
                    ),
                ))
            self.run_parallel("ablation-HPO" if ablation else "main HPO", jobs)
        else:
            while True:
                jobs = []
                counts_by_context = {ctx: self.hpo_status(ctx) for ctx in pending_contexts}
                planned_by_context = {ctx: 0 for ctx in pending_contexts}
                while len(jobs) < len(self.gpus):
                    added = False
                    for ctx in pending_contexts:
                        counts = counts_by_context[ctx]
                        planned = planned_by_context[ctx]
                        if counts["complete"] + counts["running"] + planned >= target_completed:
                            continue
                        if counts["attempted"] + planned >= max_attempted:
                            continue
                        stage = "ablation-hpo" if ctx.is_ablation else "hpo"
                        jobs.append(Job(
                            name=(
                                f"{ctx.label} {stage} trial_job "
                                f"{counts['attempted'] + planned + 1}/{max_attempted}"
                            ),
                            cmd=self.runner_cmd(
                                stage,
                                ctx,
                                *self.optuna_args(target_completed, max_attempted, n_trials=1),
                            ),
                        ))
                        planned_by_context[ctx] = planned + 1
                        added = True
                        if len(jobs) >= len(self.gpus):
                            break
                    if not added:
                        break
                if not jobs:
                    break
                self.run_parallel("ablation-HPO batch" if ablation else "main HPO batch", jobs)
        for ctx in pending_contexts:
            stage = "ablation-hpo" if ctx.is_ablation else "hpo"
            self.run_command(
                f"{ctx.label} hpo summary",
                self.runner_cmd(
                    stage,
                    ctx,
                    *self.optuna_args(target_completed, max_attempted, n_trials=0),
                ),
                gpu=self.gpus[0],
            )
            self.run_command(
                f"{ctx.label} top5-reeval",
                self.runner_cmd(
                    "top5-reeval",
                    ctx,
                    "--storage",
                    self.storage,
                ),
                gpu=self.gpus[0],
            )

    def selected_config_for_evaluation(self, ctx: RunContext) -> Path:
        if not ctx.is_ablation or ctx.hpo_required:
            return ctx.selected_config
        return self.ctmp_context().selected_config

    def evaluation_jobs_for_context(self, ctx: RunContext) -> list[Job]:
        if self.artifact_done(ctx.evaluation_summary):
            return []
        selected = self.selected_config_for_evaluation(ctx)
        if not selected.exists() and not self.dry_run:
            raise SystemExit(f"selected config missing for {ctx.label}: {selected}")
        jobs = []
        split_ids = read_eval_split_ids(ctx.run_dir) if (ctx.run_dir / "d_eval_split_artifact.json").exists() else expected_eval_split_ids()
        for split_id in split_ids:
            split_path = ctx.run_dir / "evaluation" / f"{split_id}.json"
            if self.artifact_done(split_path):
                continue
            stage = "ablation-evaluate" if ctx.is_ablation else "evaluate"
            jobs.append(Job(
                name=f"{ctx.label} evaluate {split_id}",
                cmd=self.runner_cmd(
                    stage,
                    ctx,
                    "--selected-config",
                    selected,
                    "--eval-split-id",
                    split_id,
                    "--no-summary",
                ),
            ))
        return jobs

    def evaluate_stage(self, contexts: list[RunContext], stage_name: str) -> None:
        jobs = []
        pending_contexts = []
        for ctx in contexts:
            ctx_jobs = self.evaluation_jobs_for_context(ctx)
            jobs.extend(ctx_jobs)
            if ctx_jobs or not self.artifact_done(ctx.evaluation_summary):
                pending_contexts.append(ctx)
        self.run_parallel(stage_name, jobs)
        for ctx in pending_contexts:
            self.run_command(
                f"{ctx.label} finalize-evaluate",
                self.runner_cmd(
                    "finalize-evaluate",
                    ctx,
                    include_root=False,
                ),
            )

    def pair_and_analyze(self, main_contexts: list[RunContext], ablation_contexts: list[RunContext]) -> None:
        ctmp = self.ctmp_context()
        summary_by_key = {ctx.key: ctx.evaluation_summary for ctx in main_contexts}
        comparisons = [
            f"F1,ctmp_gin,a3tgcn_2_points,{summary_by_key['ctmp_gin']},{summary_by_key['a3tgcn_2_points']}",
            f"F2,ctmp_gin,gin,{summary_by_key['ctmp_gin']},{summary_by_key['gin']}",
            f"F2,ctmp_gin,gin_gru_2_points,{summary_by_key['ctmp_gin']},{summary_by_key['gin_gru_2_points']}",
        ]
        for ctx in ablation_contexts:
            comparisons.append(f"F3,ctmp_gin,{ctx.label},{ctmp.evaluation_summary},{ctx.evaluation_summary}")
            if self.args.sesoi is not None:
                comparisons.append(f"F4,{ctx.label},ctmp_gin,{ctx.evaluation_summary},{ctmp.evaluation_summary}")

        paired_path = self.run_dir / "paired_results.json"
        analysis_path = self.run_dir / "statistical_analysis.json"
        if self.artifact_done(paired_path) and self.artifact_done(analysis_path):
            self.notifier.send("[STAGE_SKIP] pair-results/analyze already complete")
            return
        cmd = [
            self.python_bin,
            "-m",
            "src.protocol.runner",
            "--stage",
            "pair-results",
            "--run-dir",
            str(self.run_dir),
            "--split-artifact",
            str(ctmp.run_dir / "d_eval_split_artifact.json"),
            "--paired-results",
            str(paired_path),
        ]
        for comparison in comparisons:
            cmd.extend(["--comparison", comparison])
        if not self.artifact_done(paired_path):
            self.run_command("pair-results", cmd)
        else:
            self.notifier.send("[STAGE_SKIP] pair-results already complete")

        analyze_cmd = [
            self.python_bin,
            "-m",
            "src.protocol.runner",
            "--stage",
            "analyze",
            "--run-dir",
            str(self.run_dir),
            "--paired-results",
            str(paired_path),
        ]
        if self.args.sesoi is not None:
            analyze_cmd.extend(["--sesoi", str(self.args.sesoi)])
        if not self.artifact_done(analysis_path):
            self.run_command("analyze", analyze_cmd)
        else:
            self.notifier.send("[STAGE_SKIP] analyze already complete")

    def write_plan(self, main_contexts: list[RunContext], ablation_contexts: list[RunContext]) -> None:
        payload = {
            "run_dir": str(self.run_dir),
            "root": str(self.root),
            "gpus": self.gpus,
            "storage_backend": storage_backend(self.storage),
            "main_models": [ctx.key for ctx in main_contexts],
            "ablation_variants": [ctx.variant for ctx in ablation_contexts],
            "hpo_trials": self.args.hpo_trials,
            "max_hpo_attempts": self.args.max_hpo_attempts,
            "ablation_hpo_trials": self.args.ablation_hpo_trials,
            "ablation_max_hpo_attempts": self.args.ablation_max_hpo_attempts,
        }
        if self.dry_run:
            print(json.dumps(payload, indent=2))
            return
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "pipeline_plan.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def run(self) -> None:
        self.validate()
        main_contexts = self.main_contexts()
        ablation_contexts = self.ablation_contexts()
        all_contexts = main_contexts + ablation_contexts
        self.write_plan(main_contexts, ablation_contexts)
        self.notifier.send(
            f"[PIPELINE_START] run_dir={self.run_dir} gpus={','.join(self.gpus)} "
            f"main={len(main_contexts)} ablations={len(ablation_contexts)} "
            f"resume={'on' if self.resume else 'off'} next_log={self.job_counter + 1:04d} "
            f"xgboost=excluded"
        )
        self.prepare_and_preflight(all_contexts)
        self.hpo_stage(main_contexts, ablation=False)
        self.evaluate_stage(main_contexts, "main evaluate")
        independent_ablations = [ctx for ctx in ablation_contexts if ctx.hpo_required]
        inherited_ablations = [ctx for ctx in ablation_contexts if not ctx.hpo_required]
        self.hpo_stage(independent_ablations, ablation=True)
        self.evaluate_stage(independent_ablations + inherited_ablations, "ablation evaluate")
        self.pair_and_analyze(main_contexts, ablation_contexts)
        self.notifier.send(f"[PIPELINE_DONE] run_dir={self.run_dir} analysis={self.run_dir / 'statistical_analysis.json'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the protocol pipeline with one single-GPU process per GPU.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--root", default="src/data")
    parser.add_argument("--gpus", default="auto", help="'auto' or comma-separated GPU ids, e.g. 0,1,2,3")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--storage", default=None, help="PostgreSQL Optuna storage URL. Defaults to PROTOCOL_OPTUNA_STORAGE.")
    parser.add_argument("--codebook", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--notify-trials", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--discord-bot-name", default="protocol_multigpu")
    parser.add_argument("--hpo-trials", type=int, default=40)
    parser.add_argument("--max-hpo-attempts", type=int, default=80)
    parser.add_argument("--ablation-hpo-trials", type=int, default=40)
    parser.add_argument("--ablation-max-hpo-attempts", type=int, default=80)
    parser.add_argument("--ablation-variants", default=",".join(DEFAULT_ABLATION_VARIANTS))
    parser.add_argument("--sesoi", type=float, default=None)
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("--omp-threads", type=int, default=4)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    ProtocolPipeline(args).run()


if __name__ == "__main__":
    main()
