# trial-conversion-model

Predicts, from a trial's first 3 days of behavior, whether the trial will convert to a paid plan at the end of day 14, so the growth team can reach trials that look unlikely to convert while they are still live. The business case and rollout plan are in the model plan document.

## Setup

Install the dependencies and the package:

```
uv sync
```

The training data is not committed to the repository. Copy `.env.example` to `.env` and replace the password placeholder with the one from the course's Tools & Setup lesson, then materialize the extract:

```
uv run scripts/fetch_data.py
```

This writes `data/01_raw/trial_snapshot.csv`. Training always runs from that file, never from the live table, so the training data cannot shift between runs.

## Train

```
uv run scripts/train.py
```

This builds the processed training table from the raw extract, trains the model, and writes `models/model.json` and `models/metrics.json`.

## Serve

The model is served over HTTP so the teams that need scores can get them on demand. Run the service locally:

```
uv run uvicorn trial_conversion_model.api.main:app --reload
```

Or build and run it as a container, which is how it ships:

```
docker build -t trial-conversion-model .
docker run -p 8000:8000 trial-conversion-model
```

`POST /predict` takes one trial's first-3-day base aggregates and returns its conversion probability plus a low/medium/high band; `GET /health` reports service status. Interactive docs live at `/docs` while the service runs.

### Checking it works

With the service running, three requests and what each should come back with.

A steady trial, spread across the first three days:

```
curl -s -X POST localhost:8000/predict -H "Content-Type: application/json" -d '{
  "sessions_day1": 3, "sessions_day2": 2, "sessions_day3": 2,
  "listen_sessions_3d": 5, "total_minutes_3d": 180,
  "country": "US", "device_type": "iOS"
}'
```

```
{"conversion_probability":0.8593,"conversion_band":"high"}
```

The same trial's activity crammed into day one:

```
curl -s -X POST localhost:8000/predict -H "Content-Type: application/json" -d '{
  "sessions_day1": 9, "sessions_day2": 0, "sessions_day3": 0,
  "listen_sessions_3d": 4, "total_minutes_3d": 150,
  "country": "US", "device_type": "iOS"
}'
```

```
{"conversion_probability":0.3385,"conversion_band":"medium"}
```

A request with `sessions_day1` missing, which the contract turns down before any
of your code runs:

```
curl -i -s -X POST localhost:8000/predict -H "Content-Type: application/json" -d '{
  "sessions_day2": 0, "sessions_day3": 0,
  "listen_sessions_3d": 4, "total_minutes_3d": 150,
  "country": "US", "device_type": "iOS"
}'
```

```
HTTP/1.1 422 Unprocessable Entity
```

## Monitor

Two jobs run daily: `scripts/score.py` publishes the ranked list and `scripts/check_drift.py` compares the newest cohort against the training extract. Both write to the artifacts bucket, under `predictions/` and `monitoring/`.

A cohort counts as drifted when at least half of the eight input features drift. On a drifted verdict, `scripts/check_drift.py` posts the result to the Slack webhook in `SLACK_WEBHOOK_URL`. Put that variable in `.env` locally and in `/home/ubuntu/.env` on the server, where the cron containers pick it up through `--env-file`. Alerting is best effort: an unset webhook or a Slack outage never stops the check or the report.

## Dashboard

`dashboard.py` reads the published artifacts and shows the latest ranked list, the latest drift verdict with a link to the Evidently report, the model's holdout metrics, and how predictions and drift have moved over the past month:

```
uv run streamlit run dashboard.py
```

The page reloads every five minutes, so the daily jobs show up without a restart. On a fresh bucket, run `uv run scripts/backfill.py` once to give the charts a month of history.

## Layout

- `dashboard.py`: the Streamlit operations dashboard; it reads the published artifacts and writes nothing.
- `src/trial_conversion_model/`: the package. `data.py` acquires the extract from the database and loads the pipeline's inputs; `features.py` derives the model features from the snapshot's base aggregates and writes the processed training table; `train.py` trains, evaluates, and saves the model; `predict.py` scores trials from their base aggregates; `api/` is the FastAPI service (`main.py` builds the app, `routes.py` holds the endpoints, `schemas.py` defines the request and response shapes).
- `scripts/`: thin entry points that call into the package (`fetch_data.py` materializes the extract, `train.py` builds the training table and trains). Production runs these; the logic stays importable and testable in `src/`.
- `notebooks/`: exploration only. Notebooks import from the package; no pipeline logic lives here.
- `data/01_raw/`: the immutable extract as pulled from the database (never committed, never modified).
- `data/02_interim/`: reserved for intermediate outputs in multi-step pipelines; this project goes straight from raw to processed, so it stays empty.
- `data/03_processed/`: the model-ready training table written by the pipeline (never committed).
- `models/`: trained model artifacts and metrics (not committed).
