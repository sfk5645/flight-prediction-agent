"""CLI entrypoint: ingest, transform, train, serve, agent, ui."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Optional

import typer
import uvicorn

from flight_agent.config import ROOT, ensure_dirs, get_settings, load_project_config

app = typer.Typer(
    name="flight",
    help="Flight delay prediction agent — ingest, train, serve, chat.",
    no_args_is_help=True,
)
ingest_app = typer.Typer(help="Data ingest commands")
app.add_typer(ingest_app, name="ingest")


@ingest_app.command("all")
def ingest_all(
    sample: bool = typer.Option(
        False,
        "--sample",
        help="Generate synthetic BTS sample instead of downloading (demo/CI).",
    ),
    to_r2: bool = typer.Option(
        False,
        "--to-r2",
        help="Upload each Parquet object directly to Cloudflare R2 during ingest.",
    ),
    keep_local: bool = typer.Option(
        True,
        "--keep-local/--no-keep-local",
        help="Keep a local copy under data/raw (default: keep). Use --no-keep-local with --to-r2.",
    ),
    sync_r2: bool = typer.Option(
        False,
        "--sync-r2",
        help="After local ingest, bulk-upload data/raw to R2 (legacy). Prefer --to-r2.",
    ),
    incremental: bool = typer.Option(
        True,
        "--incremental/--full",
        help="Skip months already in the lake (default). Use --full to re-download the window.",
    ),
    rolling: Optional[bool] = typer.Option(
        None,
        "--rolling/--fixed",
        help="Override date_range.mode: rolling discovers latest BTS month; "
        "fixed uses start/end in configs/project.yaml.",
    ),
    prune: bool = typer.Option(
        True,
        "--prune/--no-prune",
        help="Apply retention window (default 48 months) after ingest.",
    ),
    retrain: Optional[bool] = typer.Option(
        None,
        "--retrain/--no-retrain",
        help="After ingest, run dbt + train when the sliding window moves / new months land. "
        "Default: configs/project.yaml model.auto_retrain (true).",
    ),
    train_sample_limit: Optional[int] = typer.Option(
        None,
        help="Optional row cap passed to auto-retrain.",
    ),
) -> None:
    """Run airports + weather + BTS ingest (rolling + incremental by default)."""
    from flight_agent.ingest.airports import ingest_airports
    from flight_agent.ingest.bts import ingest_bts
    from flight_agent.ingest.weather import ingest_weather
    from flight_agent.train.retrain import (
        auto_retrain_enabled,
        mark_trained,
        should_auto_retrain,
    )

    if to_r2 and not get_settings().r2_configured:
        typer.echo("ERROR: --to-r2 requires R2_* credentials in .env", err=True)
        raise typer.Exit(1)
    if not keep_local and not to_r2:
        typer.echo("ERROR: --no-keep-local requires --to-r2 (nowhere else to write).", err=True)
        raise typer.Exit(1)

    do_retrain = auto_retrain_enabled() if retrain is None else retrain

    ensure_dirs()
    ingest_airports(to_r2=to_r2, keep_local=keep_local)
    ingest_weather(
        use_sample=sample,
        to_r2=to_r2,
        keep_local=keep_local,
        incremental=incremental,
        rolling=rolling,
    )
    bts_result = ingest_bts(
        use_sample=sample,
        to_r2=to_r2,
        keep_local=keep_local,
        incremental=incremental,
        rolling=rolling,
    )
    if sync_r2 and not to_r2:
        from flight_agent.ingest.r2_sync import sync_to_r2

        sync_to_r2(prefix="raw/")

    pruned_bts = False
    if prune:
        from flight_agent.ingest.retention import prune_all

        result = prune_all(include_r2=to_r2 or sync_r2 or get_settings().r2_configured)
        pruned_bts = bool(result.local_bts_removed or result.r2_bts_objects_removed)
        typer.echo(
            f"Prune done (cutoff={result.cutoff}): "
            f"bts_partitions={len(result.local_bts_removed)}, "
            f"weather_rows={result.local_weather_rows_removed}, "
            f"r2_bts={result.r2_bts_objects_removed}, "
            f"r2_weather_rows={result.r2_weather_rows_removed}, "
            f"duckdb_rows={result.duckdb_rows_removed}"
        )
    typer.echo("Ingest complete.")

    if not do_retrain:
        return

    needed, reason = should_auto_retrain(
        bts_result.window,
        new_bts_months=len(bts_result.months_written),
        pruned_bts=pruned_bts,
    )
    if not needed:
        typer.echo(f"Auto-retrain skipped ({reason}).")
        return

    typer.echo(f"Auto-retrain triggered ({reason}) → dbt build + train…")
    from_r2 = bool(to_r2)
    code = _run_dbt_build(from_r2=from_r2)
    if code != 0:
        typer.echo("ERROR: dbt build failed; skipping train.", err=True)
        raise typer.Exit(code)

    from flight_agent.train.train import train_model

    metrics = train_model(sample_limit=train_sample_limit, publish_hf=False)
    mark_trained(bts_result.window, reason=reason)
    typer.echo(f"Auto-retrain complete: {metrics}")


@ingest_app.command("airports")
def ingest_airports_cmd(
    to_r2: bool = typer.Option(False, "--to-r2"),
    keep_local: bool = typer.Option(True, "--keep-local/--no-keep-local"),
) -> None:
    from flight_agent.ingest.airports import ingest_airports

    ensure_dirs()
    path = ingest_airports(to_r2=to_r2, keep_local=keep_local)
    typer.echo(f"Wrote {path}")


@ingest_app.command("weather")
def ingest_weather_cmd(
    sample: bool = typer.Option(False, "--sample", help="Use synthetic weather."),
    to_r2: bool = typer.Option(False, "--to-r2"),
    keep_local: bool = typer.Option(True, "--keep-local/--no-keep-local"),
    incremental: bool = typer.Option(True, "--incremental/--full"),
    rolling: Optional[bool] = typer.Option(None, "--rolling/--fixed"),
) -> None:
    from flight_agent.ingest.weather import ingest_weather

    ensure_dirs()
    paths = ingest_weather(
        use_sample=sample,
        to_r2=to_r2,
        keep_local=keep_local,
        incremental=incremental,
        rolling=rolling,
    )
    typer.echo(f"Wrote {len(paths)} weather files")


@ingest_app.command("bts")
def ingest_bts_cmd(
    sample: bool = typer.Option(False, "--sample", help="Use synthetic sample data."),
    to_r2: bool = typer.Option(False, "--to-r2"),
    keep_local: bool = typer.Option(True, "--keep-local/--no-keep-local"),
    incremental: bool = typer.Option(True, "--incremental/--full"),
    rolling: Optional[bool] = typer.Option(None, "--rolling/--fixed"),
) -> None:
    from flight_agent.ingest.bts import ingest_bts

    ensure_dirs()
    paths = ingest_bts(
        use_sample=sample,
        to_r2=to_r2,
        keep_local=keep_local,
        incremental=incremental,
        rolling=rolling,
    )
    typer.echo(f"Wrote {len(paths.written)} BTS parquet partitions")


@ingest_app.command("sync-r2")
def sync_r2_cmd(
    prefix: str = typer.Option("raw/", help="Local prefix under data/ to sync."),
) -> None:
    from flight_agent.ingest.r2_sync import sync_to_r2

    n = sync_to_r2(prefix=prefix)
    typer.echo(f"Uploaded {n} objects to R2")


@ingest_app.command("prune")
def prune_cmd(
    months: Optional[int] = typer.Option(
        None,
        help="Override retention months (default: configs/project.yaml retention_months).",
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be removed."),
    skip_r2: bool = typer.Option(False, "--skip-r2", help="Do not prune Cloudflare R2."),
) -> None:
    """Remove lake data older than the retention window (default 48 months)."""
    from flight_agent.ingest.retention import prune_all

    result = prune_all(months=months, dry_run=dry_run, include_r2=not skip_r2)
    typer.echo(
        f"Prune {'(dry-run) ' if dry_run else ''}complete cutoff={result.cutoff}: "
        f"local_bts={len(result.local_bts_removed)}, "
        f"local_weather_rows={result.local_weather_rows_removed}, "
        f"r2_bts={result.r2_bts_objects_removed}, "
        f"r2_weather_rows={result.r2_weather_rows_removed}, "
        f"duckdb_rows={result.duckdb_rows_removed}"
    )
    if not dry_run and (
        result.local_bts_removed
        or result.local_weather_rows_removed
        or result.r2_bts_objects_removed
        or result.r2_weather_rows_removed
        or result.duckdb_rows_removed
    ):
        typer.echo(
            "Tip: run `flight dbt build` or `flight dbt build --from-r2` "
            "to fully refresh marts from pruned Parquet."
        )


def _dbt_env(*, from_r2: bool = False) -> dict[str, str]:
    import os

    from flight_agent.ingest.storage import parquet_root_for_dbt

    settings = get_settings()
    env = {k: str(v) for k, v in os.environ.items()}
    env["FLIGHT_DATA_DIR"] = str(Path(settings.data_dir).resolve())
    env["FLIGHT_DUCKDB_PATH"] = str(Path(settings.duckdb_path).resolve())
    if from_r2:
        if not settings.r2_configured:
            raise RuntimeError("dbt --from-r2 requires R2 credentials in .env")
        # DuckDB wants host without scheme
        endpoint = settings.r2_endpoint.replace("https://", "").replace("http://", "")
        env["FLIGHT_PARQUET_ROOT"] = parquet_root_for_dbt(prefer_r2=True)
        env["FLIGHT_DBT_TARGET"] = "r2"
        env["R2_S3_ENDPOINT"] = endpoint
        env["R2_ACCESS_KEY_ID"] = settings.r2_access_key_id
        env["R2_SECRET_ACCESS_KEY"] = settings.r2_secret_access_key
    else:
        env["FLIGHT_PARQUET_ROOT"] = parquet_root_for_dbt(prefer_r2=False)
        env["FLIGHT_DBT_TARGET"] = "dev"
    return env


def _run_dbt_build(*, from_r2: bool = False) -> int:
    ensure_dirs()
    dbt_dir = ROOT / "dbt"
    try:
        env = _dbt_env(from_r2=from_r2)
    except RuntimeError as exc:
        typer.echo(f"ERROR: {exc}", err=True)
        return 1
    args = ["dbt", "build", "--project-dir", str(dbt_dir), "--profiles-dir", str(dbt_dir)]
    typer.echo(f"Running: {' '.join(args)} (parquet_root={env['FLIGHT_PARQUET_ROOT']})")
    code = subprocess.run(args, cwd=str(dbt_dir), env=env).returncode
    if code != 0:
        return code
    if from_r2 and get_settings().r2_configured:
        from flight_agent.ingest.warehouse import push_warehouse_to_r2

        typer.echo("Pushing warehouse (curated Parquet + DuckDB) to R2…")
        result = push_warehouse_to_r2()
        typer.echo(
            f"Warehouse on R2: {len(result['curated'])} curated files, "
            f"duckdb={result['duckdb']}, deleted_local={result['deleted_local']}"
        )
    return 0


@app.command("dbt")
def run_dbt(
    command: str = typer.Argument("run", help="dbt subcommand: run, test, build, docs"),
    select: Optional[str] = typer.Option(None, "--select", "-s"),
    from_r2: bool = typer.Option(
        False,
        "--from-r2",
        help="Read bronze Parquet from Cloudflare R2 instead of local data/raw.",
    ),
    push_warehouse: bool = typer.Option(
        True,
        "--push-warehouse/--no-push-warehouse",
        help="After a successful build with --from-r2, export curated marts + DuckDB to R2 "
        "and delete the local .duckdb (unless FLIGHT_DUCKDB_KEEP_LOCAL=true).",
    ),
) -> None:
    """Run dbt-duckdb against local Parquet lake (or R2 with --from-r2)."""
    ensure_dirs()
    dbt_dir = ROOT / "dbt"
    try:
        env = _dbt_env(from_r2=from_r2)
    except RuntimeError as exc:
        typer.echo(f"ERROR: {exc}", err=True)
        raise typer.Exit(1) from exc
    args = ["dbt", command, "--project-dir", str(dbt_dir), "--profiles-dir", str(dbt_dir)]
    if select:
        args.extend(["--select", select])
    typer.echo(f"Running: {' '.join(args)} (parquet_root={env['FLIGHT_PARQUET_ROOT']})")
    result = subprocess.run(args, cwd=str(dbt_dir), env=env)
    if result.returncode != 0:
        raise typer.Exit(result.returncode)
    if (
        push_warehouse
        and command == "build"
        and from_r2
        and get_settings().r2_configured
    ):
        from flight_agent.ingest.warehouse import push_warehouse_to_r2

        typer.echo("Pushing warehouse (curated Parquet + DuckDB) to R2…")
        pushed = push_warehouse_to_r2()
        typer.echo(
            f"Warehouse on R2: {len(pushed['curated'])} curated files, "
            f"duckdb={pushed['duckdb']}, deleted_local={pushed['deleted_local']}"
        )
    raise typer.Exit(0)


warehouse_app = typer.Typer(help="DuckDB / curated warehouse on R2")
app.add_typer(warehouse_app, name="warehouse")

model_app = typer.Typer(help="Trained model artifacts on R2 (for Streamlit Cloud)")
app.add_typer(model_app, name="model")


@model_app.command("push")
def model_push_cmd() -> None:
    """Upload models/local artifacts to R2 under models/."""
    from flight_agent.train.r2_model import push_model_to_r2

    uploaded = push_model_to_r2()
    typer.echo(f"Pushed {len(uploaded)} object(s) to R2")


@model_app.command("pull")
def model_pull_cmd(
    force: bool = typer.Option(False, "--force", help="Re-download even if local model exists."),
) -> None:
    """Download model artifacts from R2 into models/local."""
    from flight_agent.train.r2_model import pull_model_from_r2

    path = pull_model_from_r2(force=force)
    typer.echo(f"Model ready at {path}")


@warehouse_app.command("push")
def warehouse_push_cmd(
    keep_local: bool = typer.Option(
        False,
        "--keep-local/--delete-local",
        help="Keep the local .duckdb after upload (default: delete to free disk).",
    ),
) -> None:
    """Export curated marts + upload DuckDB to R2, then optionally delete local DB."""
    from flight_agent.ingest.warehouse import push_warehouse_to_r2

    if not get_settings().r2_configured:
        typer.echo("ERROR: R2 credentials required.", err=True)
        raise typer.Exit(1)
    result = push_warehouse_to_r2(delete_local=not keep_local)
    typer.echo(
        f"Pushed curated={len(result['curated'])} duckdb={result['duckdb']} "
        f"deleted_local={result['deleted_local']}"
    )


@warehouse_app.command("pull")
def warehouse_pull_cmd(
    force: bool = typer.Option(False, "--force", help="Overwrite local DuckDB if present."),
) -> None:
    """Download warehouse/flight_agent.duckdb from R2 (needs local disk space)."""
    from flight_agent.ingest.warehouse import download_duckdb_from_r2

    path = download_duckdb_from_r2(force=force)
    typer.echo(f"Local DuckDB ready at {path}")


@app.command("train")
def train_cmd(
    sample_limit: Optional[int] = typer.Option(
        None,
        help="Rows to sample from the warehouse (default: model.train_rows in project.yaml). "
        "Use 0 to attempt the full lake (may OOM).",
    ),
    publish_hf: bool = typer.Option(False, "--publish-hf", help="Push model to Hugging Face Hub."),
    publish_r2: Optional[bool] = typer.Option(
        None,
        "--publish-r2/--no-publish-r2",
        help="Push model to R2 (default: on when R2_* is configured).",
    ),
) -> None:
    """Train delay classifier and log to MLflow."""
    from flight_agent.ingest.schedule import resolve_ingest_window
    from flight_agent.train.retrain import mark_trained
    from flight_agent.train.train import train_model

    ensure_dirs()
    metrics = train_model(
        sample_limit=sample_limit, publish_hf=publish_hf, publish_r2=publish_r2
    )
    window = resolve_ingest_window()
    mark_trained(window, reason="manual_train")
    typer.echo(f"Training complete: {metrics}")


@app.command("serve")
def serve_cmd(
    host: Optional[str] = None,
    port: Optional[int] = None,
) -> None:
    """Start FastAPI prediction API."""
    settings = get_settings()
    uvicorn.run(
        "flight_agent.serve.app:app",
        host=host or settings.api_host,
        port=port or settings.api_port,
        reload=False,
    )


@app.command("ui")
def ui_cmd(port: int = typer.Option(8501, help="Streamlit port")) -> None:
    """Launch Streamlit chat UI."""
    # Prefer repo-root entrypoint (same file Streamlit Community Cloud uses).
    ui_path = ROOT / "streamlit_app.py"
    if not ui_path.exists():
        ui_path = ROOT / "src" / "flight_agent" / "ui" / "app.py"
    result = subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(ui_path), "--server.port", str(port)],
    )
    raise typer.Exit(result.returncode)


@app.command("agent")
def agent_cmd(question: str = typer.Argument(..., help="Question for the ops agent")) -> None:
    """Ask the LangGraph + Groq agent a one-shot question."""
    from flight_agent.agent.graph import ask_agent

    answer = ask_agent(question)
    typer.echo(answer)


@app.command("demo")
def demo_cmd() -> None:
    """End-to-end local demo: sample ingest → dbt → train."""
    from flight_agent.ingest.airports import ingest_airports
    from flight_agent.ingest.bts import ingest_bts
    from flight_agent.ingest.weather import ingest_weather
    from flight_agent.train.train import train_model

    ensure_dirs()
    typer.echo("Ingesting sample data…")
    ingest_airports()
    ingest_weather(use_sample=True, rolling=False, incremental=False)
    bts_result = ingest_bts(use_sample=True, rolling=False, incremental=False)

    dbt_dir = ROOT / "dbt"
    env = _dbt_env(from_r2=False)
    typer.echo("Running dbt build…")
    result = subprocess.run(
        ["dbt", "build", "--project-dir", str(dbt_dir), "--profiles-dir", str(dbt_dir)],
        cwd=str(dbt_dir),
        env=env,
    )
    if result.returncode != 0:
        raise typer.Exit(result.returncode)

    typer.echo("Training model…")
    from flight_agent.train.retrain import mark_trained

    metrics = train_model(sample_limit=50_000, publish_hf=False)
    mark_trained(bts_result.window, reason="demo")
    typer.echo(f"Demo complete: {metrics}")
    typer.echo("Next: `flight serve` and `flight ui`")


@app.command("drift")
def drift_cmd() -> None:
    """Optional Evidently drift report (requires flight-agent[drift])."""
    from flight_agent.train.drift import run_drift_report

    path = run_drift_report()
    typer.echo(f"Wrote drift report to {path}")


@app.command("info")
def info_cmd() -> None:
    """Print resolved paths and hub config."""
    settings = get_settings()
    cfg = load_project_config()
    typer.echo(f"root={ROOT}")
    typer.echo(f"data_dir={settings.data_dir}")
    typer.echo(f"duckdb={settings.duckdb_path}")
    typer.echo(f"hubs={cfg['hubs']}")
    typer.echo(f"r2_configured={settings.r2_configured}")
    if settings.r2_configured:
        typer.echo(f"r2_bucket=s3://{settings.r2_bucket}/raw")


if __name__ == "__main__":
    app()
