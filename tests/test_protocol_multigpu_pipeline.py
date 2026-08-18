from __future__ import annotations

import pytest

from scripts.protocol_multigpu_pipeline import (
    DEFAULT_ABLATION_VARIANTS,
    DiscordNotifier,
    ProtocolPipeline,
    build_parser,
    detect_gpus,
    existing_log_counter,
    json_artifact_complete,
    log_tail,
    require_postgresql_storage,
)


def test_detect_gpus_prefers_explicit_list() -> None:
    assert detect_gpus("0,2,7") == ["0", "2", "7"]


def test_detect_gpus_uses_visible_devices_for_auto(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,5")
    assert detect_gpus("auto") == ["3", "5"]


def test_postgresql_storage_is_required() -> None:
    assert require_postgresql_storage("postgresql+psycopg2://user:pass@host/db").startswith("postgresql")
    with pytest.raises(SystemExit, match="PostgreSQL"):
        require_postgresql_storage("sqlite:///local.db")


def test_discord_webhook_is_required_even_for_dry_run() -> None:
    notifier = DiscordNotifier(None, "test_bot", dry_run=True)
    with pytest.raises(SystemExit, match="DISCORD_WEBHOOK_URL is required"):
        notifier.require_ready()


def test_discord_ready_sends_initial_message(monkeypatch) -> None:
    calls = []

    class Response:
        status_code = 204
        text = ""

    def fake_post(url, json, timeout):
        calls.append((url, json, timeout))
        return Response()

    monkeypatch.setattr("scripts.protocol_multigpu_pipeline.requests.post", fake_post)
    notifier = DiscordNotifier("https://discord.example/webhook", "test_bot", dry_run=False)
    notifier.require_ready()
    assert calls[0][1]["username"] == "test_bot"
    assert "[PIPELINE_READY]" in calls[0][1]["content"]


def test_pipeline_defaults_exclude_xgboost_and_assign_one_visible_gpu(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://discord.example/webhook")
    args = build_parser().parse_args([
        "--run-dir",
        str(tmp_path / "run"),
        "--root",
        "src/data",
        "--gpus",
        "1,2",
        "--storage",
        "postgresql://user:pass@host/db",
        "--dry-run",
    ])
    pipeline = ProtocolPipeline(args)
    assert [ctx.key for ctx in pipeline.main_contexts()] == [
        "ctmp_gin",
        "gin",
        "a3tgcn_2_points",
        "gin_gru_2_points",
    ]
    assert "xgboost" not in [ctx.key for ctx in pipeline.main_contexts()]
    assert tuple(pipeline.ablation_variants()) == DEFAULT_ABLATION_VARIANTS
    env = pipeline.job_env("2")
    assert env["CUDA_VISIBLE_DEVICES"] == "2"
    assert "," not in env["CUDA_VISIBLE_DEVICES"]


def test_validate_requires_raw_csv(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://discord.example/webhook")
    root = tmp_path / "data"
    (root / "raw").mkdir(parents=True)
    args = build_parser().parse_args([
        "--run-dir",
        str(tmp_path / "run"),
        "--root",
        str(root),
        "--gpus",
        "0",
        "--storage",
        "postgresql://user:pass@host/db",
        "--dry-run",
    ])
    with pytest.raises(SystemExit, match="raw data file does not exist"):
        ProtocolPipeline(args).validate()


def test_log_tail_limits_output(tmp_path) -> None:
    path = tmp_path / "job.log"
    path.write_text("\n".join(f"line {index}" for index in range(10)), encoding="utf-8")
    assert log_tail(path, max_lines=3) == "line 7\nline 8\nline 9"


def test_resume_helpers_validate_json_and_continue_log_numbering(tmp_path, monkeypatch) -> None:
    valid = tmp_path / "valid.json"
    invalid = tmp_path / "invalid.json"
    valid.write_text('{"ok": true}', encoding="utf-8")
    invalid.write_text("{broken", encoding="utf-8")
    assert json_artifact_complete(valid)
    assert not json_artifact_complete(invalid)

    log_dir = tmp_path / "run" / "launcher_logs"
    log_dir.mkdir(parents=True)
    (log_dir / "0007_prepare.log").write_text("", encoding="utf-8")
    (log_dir / "0012_hpo.log").write_text("", encoding="utf-8")
    assert existing_log_counter(log_dir) == 12

    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://discord.example/webhook")
    args = build_parser().parse_args([
        "--run-dir",
        str(tmp_path / "run"),
        "--root",
        "src/data",
        "--gpus",
        "0",
        "--storage",
        "postgresql://user:pass@host/db",
        "--dry-run",
    ])
    pipeline = ProtocolPipeline(args)
    assert pipeline.job_counter == 12
    assert pipeline.artifact_done(valid)
    assert not pipeline.artifact_done(invalid)
