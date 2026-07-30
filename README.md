# Flight Delay Prediction Agent

End-to-end **data engineering + MLOps + AI agent** project for US aviation delays.

Public flight and weather data land as Parquet (local and optional Cloudflare R2),
are transformed with **dbt-duckdb**, train an **XGBoost** ≥15-minute arrival-delay
model tracked in **MLflow**, and power a **LangGraph + Ollama** ops agent with a
**Streamlit** chat UI — all on free / free-tier tooling.

## Architecture

```text
BTS + Open-Meteo + OurAirports
        │
        ▼
  Parquet lake (data/ or R2)
        │
        ▼
   dbt-duckdb marts
        │
        ├──────────────► FastAPI (/predict, /route-stats, /weather)
        ▼                              │
   XGBoost + MLflow                    ▼
        │                    LangGraph agent (Ollama)
        ▼                              │
  models/local                    Streamlit UI
```

## Stack (free-first)

| Layer | Tool |
|---|---|
| Lake | Parquet + Cloudflare R2 (optional) |
| SQL engine | DuckDB |
| Transforms | dbt-duckdb |
| Orchestration | GitHub Actions |
| ML | XGBoost + scikit-learn + MLflow |
| API | FastAPI |
| Agent | LangGraph + Ollama |
| UI | Streamlit (HF Spaces ready) |

**Hubs in scope:** LAX, JFK, ORD, DEN, ATL, IAD, DFW (see `configs/project.yaml`).

## Quick start (local demo, offline-friendly)

Requirements: Python 3.11+, [uv](https://github.com/astral-sh/uv). Optional: [Ollama](https://ollama.com) for full agent chat.

```bash
# Install
uv sync

# Sample ingest → dbt → train (no cloud accounts required)
uv run flight demo

# API
uv run flight serve
# open http://127.0.0.1:8000/docs

# Chat UI (uses Ollama if running; otherwise tool fallback)
uv run flight ui
```

One-shot agent question:

```bash
uv run flight agent "Will DL ATL to LAX on a Monday at 8am be delayed?"
```

## Data engineering

1. **Ingest** (rolling + incremental by default)
   - Discovers the latest published BTS PREZIP month, targets the last
     `retention_months` (48), and **skips months already in the lake / R2**
   - **Auto-retrain:** when new months land or the window slides, ingest runs
     `dbt build` + `train` and writes `models/local/train_state.json`
     (`model.auto_retrain: true`; disable with `--no-retrain`)
   - `flight ingest all --to-r2 --no-keep-local` — pull only missing months → R2
   - `flight ingest all --full` — re-download the whole window
   - `flight ingest all --fixed` — use static `date_range` start/end in config
   - `flight ingest all --sample` — synthetic bronze for demos/CI
   - `flight ingest prune` — drop data older than 48 months (also after `ingest all`)
   - `flight ingest weather --through-today` — daily Open-Meteo **hourly** refresh past BTS lag
2. **Transform** — `flight dbt build --from-r2` reads bronze from R2, builds marts,
   then **pushes gold to R2** (`curated/*.parquet` + `warehouse/flight_agent.duckdb`)
   and deletes the local DuckDB by default (`FLIGHT_DUCKDB_KEEP_LOCAL=false`).
   Train/serve query curated Parquet on R2 in-memory — no large local DB required.
   Manual: `flight warehouse push` / `flight warehouse pull`.
3. **Cloud lake** — set R2 vars in `.env`, then either:
   - Direct upload: `flight ingest all --to-r2 --no-keep-local`
   - Or legacy bulk sync of local files: `flight ingest sync-r2 --prefix raw/`
   Retention prune deletes R2 BTS partitions **and** trims R2 weather rows
   older than 48 months (same window as local).
4. **Schedule**
   - [`.github/workflows/daily-pipeline.yml`](.github/workflows/daily-pipeline.yml)
     — **daily** Open-Meteo **hourly** weather → R2 through today, then **dbt + retrain**
     (and model push to R2). Use workflow input `skip_retrain` to ingest only.
   - [`.github/workflows/weekly-lake-update.yml`](.github/workflows/weekly-lake-update.yml)
     — **Mondays:** incremental BTS (+ weather) → R2; auto-retrains when new BTS
     months land, weather updates, or the window moves.

Flights join weather on **airport + date + CRS hour** (origin dep hour / dest arr hour)
so delay predictions use hour-specific conditions.

## MLOps

```bash
uv run flight train
uv run flight train --publish-hf   # needs HF_TOKEN + HF_REPO_ID
```

Prefer letting ingest auto-retrain: when the rolling window advances, `flight ingest all`
runs dbt + train and records the window in `models/local/train_state.json`.
Use `--no-retrain` to ingest without training; set `model.auto_retrain: false` in
`configs/project.yaml` to turn it off by default.
Artifacts land in `models/local/` (`model.joblib`, `metrics.json`, `meta.json`)
and MLflow under `sqlite:///mlflow.db` (override with `FLIGHT_MLFLOW_TRACKING_URI`).

On macOS, if XGBoost fails to load OpenMP, install `brew install libomp` — otherwise
training automatically falls back to sklearn `HistGradientBoostingClassifier`.

Optional drift report:

```bash
uv sync --extra drift
uv run flight drift
```

## API

| Endpoint | Purpose |
|---|---|
| `GET /health` | Liveness + model loaded flag |
| `POST /predict` | Delay probability (+ congestion drivers) |
| `GET /route-stats` | Historical route delay / taxi / NAS |
| `GET /congestion` | Airport×hour congestion profile |
| `GET /carrier-stats` | Carrier reliability summary |
| `GET /weather` | Curated weather features |
| `GET /metrics` | Offline eval metrics |

## Agent

Tools (in-process): `predict_delay`, `route_stats`, `weather`, `airport_congestion`, `model_metrics`.

```bash
ollama pull llama3.2:3b
ollama serve
uv run flight ui
```

Without Ollama, the agent returns a deterministic tool summary so demos still work.

## Configuration

- [`configs/project.yaml`](configs/project.yaml) — hubs, `date_range.mode` (`rolling`|`fixed`), retention, model hyperparams
- [`.env.example`](.env.example) — R2, HF, Ollama, paths

With `date_range.mode: rolling` (default), the lake tracks the latest BTS publish
and keeps `retention_months` of history. Set `mode: fixed` for a pinned window.

## Hugging Face Space

See [`spaces/`](spaces/) for a Streamlit Space entrypoint. Publish the model with
`flight train --publish-hf` and point the Space at this repo or a slim Space copy.

## Model card (summary)

- **Target:** `arr_delay_15` — arrival delay ≥ 15 minutes (BTS definition)
- **Features:** schedule + peak/weekend, same-day bank volume, weather, historical taxi/NAS/carrier congestion profiles, route reliability
- **Model:** `OneHotEncoder` + `XGBClassifier` sklearn pipeline
- **Data:** Public BTS On-Time, Open-Meteo, OurAirports (synthetic sample available for CI)

## Project layout

```text
configs/               Hub + date + model config
dbt/                   dbt-duckdb project
src/flight_agent/
  ingest/              BTS, weather, airports, R2 sync
  features/            Training frame from DuckDB marts
  train/               MLflow training + HF publish + drift
  serve/               FastAPI + shared services
  agent/               LangGraph tools + graph
  ui/                  Streamlit chat
  cli.py               `flight` CLI
.github/workflows/     Daily pipeline + CI smoke
spaces/                HF Spaces helper
```

## License

Demo / research project. Respect BTS, Open-Meteo, and OurAirports terms when
redistributing data.
