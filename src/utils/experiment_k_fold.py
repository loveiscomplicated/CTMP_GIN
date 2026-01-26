import os
import sys
import json
import yaml
import time
import subprocess
import random
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union, List

import torch
from src.utils.experiment import (
    _now_run_id,
    _format_float,
    make_run_id,
    _get_git_info,
    _get_command_line
)

# -----------------------------
# Small utilities
# -----------------------------

def ensure_run_dir(base_dir: str, run_id: str) -> str:
    run_dir = os.path.join(base_dir, run_id)
    os.makedirs(run_dir, exist_ok=False)  # fail if collision
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "splits"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "folds"), exist_ok=True)
    return run_dir

def _atomic_write_bytes(path: str, data: bytes) -> None:
    """
    Write bytes atomically: write to tmp in same directory then os.replace.
    """
    dir_name = os.path.dirname(os.path.abspath(path))
    os.makedirs(dir_name, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", dir=dir_name)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


def save_text(path: str, text: str) -> None:
    _atomic_write_bytes(path, text.encode("utf-8"))


def save_yaml(path: str, obj: Dict[str, Any]) -> None:
    payload = yaml.safe_dump(obj, sort_keys=False, allow_unicode=True).encode("utf-8")
    _atomic_write_bytes(path, payload)


def save_json_atomic(path: str, obj: Dict[str, Any]) -> None:
    payload = json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8")
    _atomic_write_bytes(path, payload)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def append_jsonl(path: str, record: Dict[str, Any]) -> None:
    # jsonl is append-only; atomic replace isn't appropriate.
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def torch_save_atomic(path: str, obj: Any) -> None:
    """
    torch.save atomically: save to tmp then os.replace.
    """
    dir_name = os.path.dirname(os.path.abspath(path))
    os.makedirs(dir_name, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", dir=dir_name)
    os.close(fd)
    try:
        torch.save(obj, tmp_path)
        os.replace(tmp_path, path)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


def get_rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python_random_state": random.getstate(),
        "torch_rng_state": torch.get_rng_state(),
    }
    try:
        import numpy as np  # local import (optional dependency)

        state["numpy_random_state"] = np.random.get_state()
    except Exception:
        state["numpy_random_state"] = None

    if torch.cuda.is_available():
        try:
            state["torch_cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
        except Exception:
            state["torch_cuda_rng_state_all"] = None
    else:
        state["torch_cuda_rng_state_all"] = None

    return state


def set_rng_state(state: Dict[str, Any]) -> None:
    if not state:
        return

    py_state = state.get("python_random_state", None)
    if py_state is not None:
        try:
            random.setstate(py_state)
        except Exception:
            pass

    t_state = state.get("torch_rng_state", None)
    if t_state is not None:
        try:
            torch.set_rng_state(t_state)
        except Exception:
            pass

    np_state = state.get("numpy_random_state", None)
    if np_state is not None:
        try:
            import numpy as np

            np.random.set_state(np_state)
        except Exception:
            pass

    cuda_all = state.get("torch_cuda_rng_state_all", None)
    if cuda_all is not None and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(cuda_all)
        except Exception:
            pass


# -----------------------------
# Policies
# -----------------------------
@dataclass
class CheckpointPolicy_cv:
    save_ckpt: bool = True
    save_every: int = 1  # save every N epochs (0 or <0 disables periodic saving)
    save_best: bool = True  # save best checkpoint
    monitor: str = "valid_loss"  # metric name to monitor
    mode: str = "min"  # "min" for loss, "max" for auc/f1
    keep_last: bool = True  # keep "last.pt"
    save_rng_state: bool = True  # include RNG states
    save_amp_scaler: bool = True  # include GradScaler state if provided


# -----------------------------
# Logger with K-fold + resume
# -----------------------------
class ExperimentLogger_cv:
    """
    - Creates run directory + basic artifacts:
      config.final.yaml, command.txt, git.txt, metrics.jsonl
    - Checkpoints (atomic): last.pt / best.pt / epoch_k.pt
    - K-fold helpers:
      exp_state.json, folds/fold_{k}/fold_state.json, splits/fold_{k}.json
    """

    def __init__(self, cfg: Dict[str, Any], run_dir: str):
        self.cfg = cfg
        self.run_dir = run_dir

        self.metrics_path = os.path.join(run_dir, "metrics.jsonl")
        self.ckpt_dir = os.path.join(run_dir, "checkpoints")
        self.splits_dir = os.path.join(run_dir, "splits")
        self.folds_dir = os.path.join(run_dir, "folds")
        self.exp_state_path = os.path.join(run_dir, "exp_state.json")

        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.splits_dir, exist_ok=True)
        os.makedirs(self.folds_dir, exist_ok=True)

        train_cfg = cfg.get("train", {})
        self.policy = CheckpointPolicy_cv(
            save_ckpt=bool(train_cfg.get("save_ckpt", True)),
            save_every=int(train_cfg.get("save_every", 1)),
            save_best=bool(train_cfg.get("save_best", True)),
            monitor=str(train_cfg.get("monitor", "valid_loss")),
            mode=str(train_cfg.get("mode", "min")).lower(),
            keep_last=bool(train_cfg.get("keep_last", True)),
            save_rng_state=bool(train_cfg.get("save_rng_state", True)),
            save_amp_scaler=bool(train_cfg.get("save_amp_scaler", True)),
        )

        self.best_value: Optional[float] = None
        self.best_epoch: Optional[int] = None

        # Save run artifacts immediately
        save_yaml(os.path.join(run_dir, "config.final.yaml"), cfg)
        save_text(os.path.join(run_dir, "command.txt"), _get_command_line() + "\n")
        save_text(os.path.join(run_dir, "git.txt"), _get_git_info())

    # -------------------------
    # K-fold directory helpers
    # -------------------------
    def fold_dir(self, fold_id: int) -> str:
        d = os.path.join(self.folds_dir, f"fold_{fold_id}")
        os.makedirs(d, exist_ok=True)
        os.makedirs(os.path.join(d, "checkpoints"), exist_ok=True)
        return d

    def fold_ckpt_dir(self, fold_id: int) -> str:
        d = os.path.join(self.fold_dir(fold_id), "checkpoints")
        os.makedirs(d, exist_ok=True)
        return d

    def fold_state_path(self, fold_id: int) -> str:
        return os.path.join(self.fold_dir(fold_id), "fold_state.json")

    def splits_path(self, fold_id: int) -> str:
        return os.path.join(self.splits_dir, f"fold_{fold_id}.json")

    # -------------------------
    # Metrics
    # -------------------------
    def log_metrics(self, epoch: int, metrics: Dict[str, Any], fold_id: Optional[int] = None) -> None:
        record = {"epoch": epoch, **metrics}
        if fold_id is not None:
            record["fold_id"] = fold_id
        append_jsonl(self.metrics_path, record)

    # -------------------------
    # exp_state.json
    # -------------------------
    def init_exp_state_if_missing(
        self,
        n_folds: int,
        cfg_hash: Optional[str] = None,
        data_version: Optional[str] = None,
    ) -> Dict[str, Any]:
        if os.path.exists(self.exp_state_path):
            return self.load_exp_state()

        folds = {}
        for k in range(n_folds):
            folds[str(k)] = {
                "status": "not_started",  # not_started|running|done
                "best_score": None,
                "best_epoch": None,
            }

        train_cfg = self.cfg.get("train", {})
        seed = train_cfg.get("seed", None)

        state = {
            "status": "running",
            "current_fold": 0,
            "folds": folds,
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "cfg_hash": cfg_hash,
            "data_version": data_version,
            "seed": seed,
        }
        save_json_atomic(self.exp_state_path, state)
        return state

    def load_exp_state(self) -> Dict[str, Any]:
        if not os.path.exists(self.exp_state_path):
            raise FileNotFoundError(f"exp_state.json not found: {self.exp_state_path}")
        return load_json(self.exp_state_path)

    def save_exp_state(self, state: Dict[str, Any]) -> None:
        state = dict(state)
        state["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        save_json_atomic(self.exp_state_path, state)

    def mark_fold_status(
        self,
        fold_id: int,
        status: str,
        best_score: Optional[float] = None,
        best_epoch: Optional[int] = None,
    ) -> None:
        exp = self.load_exp_state()
        f = exp.get("folds", {})
        key = str(fold_id)
        if key not in f:
            f[key] = {"status": "not_started", "best_score": None, "best_epoch": None}

        f[key]["status"] = status
        if best_score is not None:
            f[key]["best_score"] = best_score
        if best_epoch is not None:
            f[key]["best_epoch"] = best_epoch

        exp["folds"] = f
        exp["current_fold"] = fold_id
        # If all done, mark experiment done
        if all(v.get("status") == "done" for v in f.values()):
            exp["status"] = "done"
        self.save_exp_state(exp)

    def get_next_fold_to_run(self) -> Optional[int]:
        """
        Priority:
        1) any fold with status=running (resume)
        2) first fold with status=not_started
        3) None if all done
        """
        exp = self.load_exp_state()
        folds = exp.get("folds", {})
        if not folds:
            return None

        # resume running first
        for k, v in folds.items():
            if v.get("status") == "running":
                return int(k)

        # then not started
        for k, v in folds.items():
            if v.get("status") == "not_started":
                return int(k)

        return None

    # -------------------------
    # fold_state.json
    # -------------------------
    def init_fold_state_if_missing(self, fold_id: int) -> Dict[str, Any]:
        path = self.fold_state_path(fold_id)
        if os.path.exists(path):
            return self.load_fold_state(fold_id)

        st = {
            "fold_id": fold_id,
            "epoch": 0,  # next epoch to run
            "global_step": 0,
            "best_score": None,
            "best_epoch": None,
            "last_ckpt_path": os.path.join(self.fold_ckpt_dir(fold_id), "last.pt"),
            "best_ckpt_path": os.path.join(self.fold_ckpt_dir(fold_id), "best.pt"),
            "rng_state_saved": False,
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        save_json_atomic(path, st)
        return st

    def load_fold_state(self, fold_id: int) -> Dict[str, Any]:
        path = self.fold_state_path(fold_id)
        if not os.path.exists(path):
            raise FileNotFoundError(f"fold_state.json not found: {path}")
        return load_json(path)

    def save_fold_state(self, fold_id: int, state: Dict[str, Any]) -> None:
        st = dict(state)
        st["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        save_json_atomic(self.fold_state_path(fold_id), st)

    def update_fold_progress(
        self,
        fold_id: int,
        next_epoch: int,
        global_step: int,
        best_score: Optional[float],
        best_epoch: Optional[int],
        rng_state_saved: bool,
    ) -> None:
        st = self.load_fold_state(fold_id)
        st["epoch"] = int(next_epoch)
        st["global_step"] = int(global_step)
        st["best_score"] = best_score
        st["best_epoch"] = best_epoch
        st["rng_state_saved"] = bool(rng_state_saved)
        self.save_fold_state(fold_id, st)

    # -------------------------
    # Split save/load
    # -------------------------
    def save_splits(
        self,
        fold_id: int,
        train_idx: List[int],
        val_idx: List[int],
        test_idx: List[int],
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        payload = {
            "fold_id": fold_id,
            "train_idx": list(map(int, train_idx)),
            "val_idx": list(map(int, val_idx)),
            "test_idx": list(map(int, test_idx)),
            "meta": meta or {},
        }
        save_json_atomic(self.splits_path(fold_id), payload)

    def load_splits(self, fold_id: int) -> Dict[str, Any]:
        path = self.splits_path(fold_id)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Split file not found: {path}")
        return load_json(path)

    # -------------------------
    # Checkpoint helpers
    # -------------------------
    def _is_better(self, value: float) -> bool:
        if self.best_value is None:
            return True
        if self.policy.mode == "min":
            return value < self.best_value
        if self.policy.mode == "max":
            return value > self.best_value
        raise ValueError(f"Unknown mode: {self.policy.mode}")

    def maybe_save_checkpoint(
        self,
        epoch: int,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        scheduler: Optional[Any],
        metrics: Dict[str, Any],
        extra: Optional[Dict[str, Any]] = None,
        fold_id: Optional[int] = None,
        global_step: Optional[int] = None,
        amp_scaler: Optional[Any] = None,  # torch.cuda.amp.GradScaler
        save_dir_override: Optional[str] = None,
    ) -> Tuple[bool, Optional[float]]:
        """
        Save checkpoints according to policy (atomic):
        - best.pt when monitored metric improves (if save_best)
        - last.pt every epoch (if keep_last)
        - epoch_{k}.pt every save_every epochs (if save_every > 0)

        Returns:
          (best_updated, best_value)
        """
        if not self.policy.save_ckpt:
            return (False, self.best_value)

        extra = extra or {}

        # directory selection
        if save_dir_override is not None:
            ckpt_dir = save_dir_override
            os.makedirs(ckpt_dir, exist_ok=True)
        else:
            ckpt_dir = self.fold_ckpt_dir(fold_id) if fold_id is not None else self.ckpt_dir

        state: Dict[str, Any] = {
            "epoch": int(epoch),
            "global_step": int(global_step) if global_step is not None else None,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "metrics": dict(metrics),
            "cfg": self.cfg,
            "fold_id": int(fold_id) if fold_id is not None else None,
            **extra,
        }

        # AMP scaler
        if self.policy.save_amp_scaler and amp_scaler is not None:
            try:
                state["amp_scaler_state_dict"] = amp_scaler.state_dict()
            except Exception:
                state["amp_scaler_state_dict"] = None
        else:
            state["amp_scaler_state_dict"] = None

        # RNG
        rng_saved = False
        if self.policy.save_rng_state:
            try:
                state["rng_state"] = get_rng_state()
                rng_saved = True
            except Exception:
                state["rng_state"] = None
                rng_saved = False
        else:
            state["rng_state"] = None
            rng_saved = False

        # keep last
        if self.policy.keep_last:
            torch_save_atomic(os.path.join(ckpt_dir, "last.pt"), state)

        # periodic
        if self.policy.save_every and self.policy.save_every > 0:
            if epoch % self.policy.save_every == 0:
                torch_save_atomic(os.path.join(ckpt_dir, f"epoch_{epoch}.pt"), state)

        best_updated = False

        # best
        if self.policy.save_best and self.policy.monitor in metrics:
            try:
                cur = float(metrics[self.policy.monitor])
            except Exception:
                cur = None

            if cur is not None and self._is_better(cur):
                self.best_value = cur
                self.best_epoch = epoch
                torch_save_atomic(os.path.join(ckpt_dir, "best.pt"), state)
                save_text(
                    os.path.join(self.run_dir, "best.txt") if fold_id is None else os.path.join(self.fold_dir(fold_id), "best.txt"),
                    f"best_epoch: {epoch}\n{self.policy.monitor}: {cur}\n",
                )
                best_updated = True

        # fold_state update hook (optional)
        if fold_id is not None:
            # Keep fold_state in sync with what we just saved.
            st = self.init_fold_state_if_missing(fold_id)
            st["best_score"] = self.best_value
            st["best_epoch"] = self.best_epoch
            st["epoch"] = int(epoch) + 1  # next epoch
            if global_step is not None:
                st["global_step"] = int(global_step)
            st["rng_state_saved"] = bool(rng_saved)
            self.save_fold_state(fold_id, st)

        return (best_updated, self.best_value)

    def load_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        ckpt_path: Optional[str] = None,
        ckpt_name: str = "last.pt",
        map_location: Optional[Union[str, torch.device]] = "cpu",
        amp_scaler: Optional[Any] = None,
        restore_rng: bool = True,
        fold_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Load checkpoint and restore model/optimizer/scheduler/(amp_scaler)/rng.
        """
        if ckpt_path is None:
            ckpt_dir = self.fold_ckpt_dir(fold_id) if fold_id is not None else self.ckpt_dir
            ckpt_path = os.path.join(ckpt_dir, ckpt_name)

        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        state = torch.load(ckpt_path, map_location=map_location)

        # 1) model
        model.load_state_dict(state["model_state_dict"])

        # 2) optimizer
        if optimizer is not None and state.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(state["optimizer_state_dict"])

        # 3) scheduler
        if scheduler is not None and state.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(state["scheduler_state_dict"])

        # 4) amp scaler
        if amp_scaler is not None and state.get("amp_scaler_state_dict") is not None:
            try:
                amp_scaler.load_state_dict(state["amp_scaler_state_dict"])
            except Exception:
                pass

        # 5) best tracking
        self.best_epoch = None
        self.best_value = None
        basename = os.path.basename(ckpt_path)
        if basename == "best.pt":
            m = state.get("metrics", {})
            if self.policy.monitor in m:
                try:
                    self.best_value = float(m[self.policy.monitor])
                except Exception:
                    self.best_value = None
            self.best_epoch = state.get("epoch", None)

        # 6) restore rng
        if restore_rng:
            try:
                rng_state = state.get("rng_state", None)
                if rng_state is not None:
                    set_rng_state(rng_state)
            except Exception:
                pass

        return state

    def resume_if_possible(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        prefer: str = "last",  # "last" or "best"
        map_location: Optional[Union[str, torch.device]] = "cpu",
        amp_scaler: Optional[Any] = None,
        restore_rng: bool = True,
        fold_id: Optional[int] = None,
    ) -> Tuple[int, int]:
        """
        Returns:
          (next_epoch, global_step)
        """
        ckpt_name = f"{prefer}.pt"
        ckpt_dir = self.fold_ckpt_dir(fold_id) if fold_id is not None else self.ckpt_dir
        ckpt_path = os.path.join(ckpt_dir, ckpt_name)

        if not os.path.exists(ckpt_path):
            return (0, 0)  # fresh start

        state = self.load_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            ckpt_path=ckpt_path,
            map_location=map_location,
            amp_scaler=amp_scaler,
            restore_rng=restore_rng,
            fold_id=fold_id,
        )
        last_epoch = int(state.get("epoch", 0))
        global_step = int(state.get("global_step", 0) or 0)
        return (last_epoch + 1, global_step)

    # -------------------------
    # Convenience: fold-aware resume using fold_state.json
    # -------------------------
    def resume_fold_if_possible(
        self,
        fold_id: int,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        map_location: Optional[Union[str, torch.device]] = "cpu",
        amp_scaler: Optional[Any] = None,
        restore_rng: bool = True,
        prefer: str = "last",
    ) -> Tuple[int, int]:
        """
        Uses fold_state.json (if exists) and last/best checkpoint to resume.
        Returns:
          (next_epoch, global_step)
        """
        self.init_fold_state_if_missing(fold_id)

        # Prefer checkpoint-based truth for safety
        next_epoch, global_step = self.resume_if_possible(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            prefer=prefer,
            map_location=map_location,
            amp_scaler=amp_scaler,
            restore_rng=restore_rng,
            fold_id=fold_id,
        )

        # Sync fold_state.json
        st = self.load_fold_state(fold_id)
        st["epoch"] = int(next_epoch)
        st["global_step"] = int(global_step)
        st["best_score"] = self.best_value
        st["best_epoch"] = self.best_epoch
        self.save_fold_state(fold_id, st)

        return next_epoch, global_step
