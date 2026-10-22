"""Operations dashboard for the trial conversion model.

Reads the artifacts the daily jobs publish to S3 — one ranked list per day
under predictions/ and one drift verdict per day under monitoring/ — and
shows the latest of each, the model's metrics, and how both have moved over
the past month.

Run it on a laptop and leave it open:

    uv run streamlit run dashboard.py
"""

import json
import os
import re
from datetime import timedelta
from io import BytesIO
from pathlib import Path

import altair as alt
import boto3
import pandas as pd
import streamlit as st
from botocore.exceptions import BotoCoreError, ClientError
from dotenv import load_dotenv

from trial_conversion_model.monitoring import DRIFT_SHARE
from trial_conversion_model.predict import BAND_LOW, to_band

load_dotenv()

PREDICTIONS_PREFIX = "predictions/"
DRIFT_PREFIX = "monitoring/"
MODEL_METRICS = Path("models/metrics.json")
MODEL_METRICS_KEY = "models/metrics.json"

# Artifacts are rewritten once a day, so a couple of minutes of cache is
# plenty; the page itself reloads every five.
CACHE_TTL = 120
REFRESH = timedelta(minutes=5)

SCORED_RE = re.compile(r"^predictions/scored_(\d{4}-\d{2}-\d{2})\.csv$")
# Real checks publish cohort_<date>.json; the one-time backfill uses
# drift_cohort_<date>.json. Both are history worth charting.
DRIFT_RE = re.compile(r"^monitoring/(?:drift_)?cohort_(\d{4}-\d{2}-\d{2})\.json$")

GREEN = "#5f8d6b"
RED = "#c25450"
BLUE = "#4c78a8"
LIGHT_BLUE = "#93b7d6"


def get_bucket() -> str:
    return os.environ.get("S3_BUCKET", "")


@st.cache_resource
def s3_client():
    return boto3.client("s3")


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def list_keys(prefix: str) -> list[str]:
    paginator = s3_client().get_paginator("list_objects_v2")
    return [
        obj["Key"]
        for page in paginator.paginate(Bucket=get_bucket(), Prefix=prefix)
        for obj in page.get("Contents", [])
    ]


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def read_csv(key: str) -> pd.DataFrame:
    body = s3_client().get_object(Bucket=get_bucket(), Key=key)["Body"].read()
    return pd.read_csv(BytesIO(body))


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def read_json(key: str) -> dict:
    body = s3_client().get_object(Bucket=get_bucket(), Key=key)["Body"].read()
    return json.loads(body)


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def prediction_history() -> pd.DataFrame:
    """One row per published ranked list: its size and how it scored."""
    rows = []
    for key in sorted(list_keys(PREDICTIONS_PREFIX)):
        match = SCORED_RE.match(key)
        if not match:
            continue
        scored = read_csv(key)
        probability = scored["conversion_probability"]
        rows.append(
            {
                "date": pd.Timestamp(match.group(1)),
                "trials": len(scored),
                "mean_probability": probability.mean(),
                "median_probability": probability.median(),
                "share_at_risk": (probability < BAND_LOW).mean(),
            }
        )
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def drift_history() -> pd.DataFrame:
    """One row per dated drift verdict, preferring a real check over backfill."""
    by_date: dict[pd.Timestamp, dict] = {}
    for key in sorted(list_keys(DRIFT_PREFIX)):
        match = DRIFT_RE.match(key)
        if not match:
            continue
        day = pd.Timestamp(match.group(1))
        backfill = "/drift_" in key
        existing = by_date.get(day)
        # A real check lands on the same day as a filled-in verdict.
        if existing is not None and existing["backfill"] and not backfill:
            continue
        verdict = read_json(key)
        drifted = bool(verdict["drift"])
        by_date[day] = {
            "date": day,
            "drifted_share": float(verdict["drifted_share"]),
            "drift": drifted,
            "verdict": "drift" if drifted else "ok",
            "backfill": backfill,
            "key": key,
        }
    history = pd.DataFrame(by_date.values())
    return history.sort_values("date").reset_index(drop=True)


@st.cache_data(ttl=600, show_spinner=False)
def report_url(key: str) -> str | None:
    """A one-hour link to the HTML report, when the check published one."""
    html_key = key.removesuffix(".json") + ".html"
    if html_key not in list_keys(DRIFT_PREFIX):
        return None
    return s3_client().generate_presigned_url(
        "get_object",
        Params={"Bucket": get_bucket(), "Key": html_key},
        ExpiresIn=3600,
    )


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def model_metrics() -> dict:
    """The trained model's metrics: local artifact first, then the bucket."""
    if MODEL_METRICS.exists():
        return json.loads(MODEL_METRICS.read_text())
    try:
        return read_json(MODEL_METRICS_KEY)
    except (BotoCoreError, ClientError):
        return {}


def latest_scored_key() -> str | None:
    keys = [key for key in list_keys(PREDICTIONS_PREFIX) if SCORED_RE.match(key)]
    # Names carry ISO dates, so the alphabetically last key is the newest.
    return max(keys) if keys else None


def date_axis() -> alt.Axis:
    return alt.Axis(format="%b %d", tickCount=8, labelAngle=0, title=None)


def share_axis() -> alt.Axis:
    return alt.Axis(format="%", title=None)


def probability_chart(history: pd.DataFrame) -> alt.Chart:
    long = history.rename(
        columns={"mean_probability": "Mean", "median_probability": "Median"}
    ).melt(
        id_vars="date",
        value_vars=["Mean", "Median"],
        var_name="Measure",
        value_name="Probability",
    )
    return (
        alt.Chart(long)
        .mark_line(point=alt.OverlayMarkDef(size=45))
        .encode(
            x=alt.X("date:T", axis=date_axis()),
            y=alt.Y("Probability:Q", axis=share_axis(), scale=alt.Scale(domain=[0, 1])),
            color=alt.Color(
                "Measure:N",
                scale=alt.Scale(domain=["Mean", "Median"], range=[BLUE, LIGHT_BLUE]),
                legend=alt.Legend(orient="top", direction="horizontal", title=None),
            ),
            tooltip=[
                alt.Tooltip("date:T", title="Date", format="%b %d, %Y"),
                alt.Tooltip("Measure:N", title="Measure"),
                alt.Tooltip("Probability:Q", title="Probability", format=".1%"),
            ],
        )
        .properties(height=250)
    )


def risk_chart(history: pd.DataFrame) -> alt.Chart:
    return (
        alt.Chart(history)
        .mark_area(
            color=alt.Gradient(
                gradient="linear",
                stops=[
                    alt.GradientStop(color="#f3d9d8", offset=0),
                    alt.GradientStop(color=RED, offset=1),
                ],
                x1=1,
                x2=1,
                y1=1,
                y2=0,
            ),
            line={"color": RED, "strokeWidth": 2},
        )
        .encode(
            x=alt.X("date:T", axis=date_axis()),
            y=alt.Y(
                "share_at_risk:Q",
                axis=share_axis(),
                scale=alt.Scale(domain=[0, 1]),
            ),
            tooltip=[
                alt.Tooltip("date:T", title="Date", format="%b %d, %Y"),
                alt.Tooltip("share_at_risk:Q", title="At risk", format=".1%"),
                alt.Tooltip("trials:Q", title="Trials"),
            ],
        )
        .properties(height=250)
    )


def drift_chart(history: pd.DataFrame) -> alt.Chart:
    bars = (
        alt.Chart(history)
        .mark_bar(size=9, cornerRadius=2)
        .encode(
            x=alt.X("date:T", axis=date_axis()),
            y=alt.Y(
                "drifted_share:Q",
                axis=share_axis(),
                scale=alt.Scale(domain=[0, 1]),
            ),
            color=alt.condition(alt.datum.drift, alt.value(RED), alt.value(GREEN)),
            tooltip=[
                alt.Tooltip("date:T", title="Date", format="%b %d, %Y"),
                alt.Tooltip("drifted_share:Q", title="Features drifted", format=".1%"),
                alt.Tooltip("verdict:N", title="Verdict"),
            ],
        )
    )
    threshold = (
        alt.Chart(pd.DataFrame({"threshold": [DRIFT_SHARE]}))
        .mark_rule(strokeDash=[5, 4], color="#8a8a8a")
        .encode(y="threshold:Q")
    )
    return (bars + threshold).properties(height=230)


def ranked_table(scored: pd.DataFrame, rows: int) -> pd.DataFrame:
    table = scored.copy()
    table["sessions_3d"] = table[
        ["sessions_day1", "sessions_day2", "sessions_day3"]
    ].sum(axis=1)
    table["band"] = table["conversion_probability"].map(to_band)
    columns = [
        "trial_id",
        "country",
        "device_type",
        "sessions_3d",
        "conversion_probability",
        "band",
    ]
    return table.head(rows)[columns]


def render() -> None:
    if not get_bucket():
        st.error(
            "`S3_BUCKET` is not set. Copy `.env.example` to `.env` and fill it in."
        )
        st.stop()

    with st.spinner("Reading the published artifacts…"):
        try:
            predictions = prediction_history()
            drift = drift_history()
            scored_key = latest_scored_key()
            scored = read_csv(scored_key) if scored_key else pd.DataFrame()
        except (BotoCoreError, ClientError) as exc:
            st.error("Could not read the artifacts bucket.")
            st.caption(str(exc))
            return

    if predictions.empty or not scored_key:
        st.info(
            "No ranked lists published yet. Run `uv run scripts/score.py` "
            "(and `uv run scripts/backfill.py` for the month of history)."
        )
        return

    latest = predictions.iloc[-1]
    st.caption(
        f"Latest artifacts: {latest['date']:%b %d, %Y}  ·  "
        f"refreshes every {REFRESH.seconds // 60} minutes"
    )

    # The four numbers a reader checks first.
    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Trials scored",
        f"{int(latest['trials']):,}",
        help="Size of the most recent day's cohort.",
        border=True,
    )
    col2.metric(
        "At risk",
        f"{latest['share_at_risk']:.0%}",
        help=(
            f"Share of the latest cohort scoring below {BAND_LOW:.0%} — "
            "the band where the lifecycle team reaches out."
        ),
        border=True,
    )
    if not drift.empty:
        verdict = drift.iloc[-1]
        col3.metric(
            "Features drifted",
            f"{verdict['drifted_share']:.0%}",
            help=(
                f"From the {verdict['date']:%b %d} check. A cohort is "
                f"flagged when {DRIFT_SHARE:.0%} of features drift."
            ),
            border=True,
        )

    left, right = st.columns([2, 1], gap="large")

    with left:
        st.subheader("Latest ranked list")
        st.caption(
            "Least likely to convert first — the lifecycle team works down "
            "this list while the trials are still live."
        )
        total = len(scored)
        rows = min(10, total)
        if total > 5:
            rows = st.slider(
                "Rows to show", min_value=5, max_value=min(50, total), value=rows
            )
        st.dataframe(
            ranked_table(scored, rows),
            hide_index=True,
            width="stretch",
            column_config={
                "trial_id": st.column_config.NumberColumn("Trial", format="%d"),
                "country": st.column_config.TextColumn("Country"),
                "device_type": st.column_config.TextColumn("Device"),
                "sessions_3d": st.column_config.NumberColumn("Sessions (3d)"),
                "conversion_probability": st.column_config.ProgressColumn(
                    "Conversion probability",
                    help="Predicted chance this trial becomes a paid plan.",
                    min_value=0.0,
                    max_value=1.0,
                    format="%.3f",
                ),
                "band": st.column_config.TextColumn("Band"),
            },
        )
        st.download_button(
            "Download the full list",
            data=scored.to_csv(index=False),
            file_name=scored_key.rsplit("/", 1)[-1],
            mime="text/csv",
        )

    with right:
        st.subheader("Latest drift check")
        if drift.empty:
            st.warning("No drift verdicts published yet.")
        else:
            verdict = drift.iloc[-1]
            with st.container(border=True):
                if verdict["drift"]:
                    st.error(
                        f"Drift detected — {verdict['drifted_share']:.0%} of features",
                        icon="🚨",
                    )
                else:
                    st.success(
                        f"No drift — {verdict['drifted_share']:.0%} of features",
                        icon="✅",
                    )
                st.caption(
                    f"Checked {verdict['date']:%b %d, %Y} · threshold "
                    f"{DRIFT_SHARE:.0%} of features"
                )
                if verdict["backfill"]:
                    st.caption("Synthetic backfill; replaced by the live check.")
                link = report_url(verdict["key"])
                if link:
                    st.link_button("Open the Evidently report", link)
                else:
                    st.caption("No HTML report for this verdict.")

        st.subheader("Model")
        metrics = model_metrics()
        if metrics:
            with st.container(border=True):
                st.metric(
                    "Holdout ROC AUC",
                    f"{metrics['test_auc']:.3f}",
                    help="Ranking quality on trials the model never saw in training.",
                )
                st.caption(
                    f"{metrics['n_train']:,} training trials · "
                    f"{metrics['n_test']:,} holdout · "
                    f"{len(metrics['features'])} encoded inputs"
                )
        else:
            st.warning("No `models/metrics.json` — train the model first.")

    st.divider()
    st.subheader("The past month")
    st.caption(
        "One point per daily cohort. Probabilities and at-risk shares come "
        "from each ranked list; drift comes from each check."
    )

    chart_left, chart_right = st.columns(2, gap="large")
    with chart_left:
        st.markdown("**Conversion probability**")
        st.altair_chart(probability_chart(predictions), width="stretch")
    with chart_right:
        st.markdown(f"**Share scoring below {BAND_LOW:.0%}**")
        st.altair_chart(risk_chart(predictions), width="stretch")

    st.markdown("**Features drifted, by day**")
    if drift.empty:
        st.caption("No drift checks published yet.")
    else:
        st.altair_chart(drift_chart(drift), width="stretch")

    with st.expander("Day by day"):
        merged = predictions.merge(
            drift[["date", "drifted_share", "verdict", "backfill"]],
            on="date",
            how="outer",
        ).sort_values("date")
        merged["source"] = merged["backfill"].map({True: "backfill", False: "live"})
        st.dataframe(
            merged[
                [
                    "date",
                    "trials",
                    "mean_probability",
                    "share_at_risk",
                    "drifted_share",
                    "verdict",
                    "source",
                ]
            ],
            hide_index=True,
            width="stretch",
            column_config={
                "date": st.column_config.DateColumn("Date", format="YYYY-MM-DD"),
                "trials": st.column_config.NumberColumn("Trials"),
                "mean_probability": st.column_config.NumberColumn(
                    "Mean probability", format="percent"
                ),
                "share_at_risk": st.column_config.NumberColumn(
                    "At risk", format="percent"
                ),
                "drifted_share": st.column_config.NumberColumn(
                    "Drifted", format="percent"
                ),
                "verdict": st.column_config.TextColumn("Verdict"),
                "source": st.column_config.TextColumn("Source"),
            },
        )

    footer_left, footer_right = st.columns([4, 1])
    footer_left.caption(f"Reading `s3://{get_bucket()}/` and `models/metrics.json`.")
    if footer_right.button("Refresh now", width="stretch"):
        st.cache_data.clear()
        st.rerun()


def main() -> None:
    st.set_page_config(
        page_title="Trial conversion · model ops",
        page_icon="📈",
        layout="wide",
    )
    st.title("Trial conversion · model ops")
    st.caption(
        "Daily ranked list, drift verdict, and model health for the "
        "trial-to-paid model."
    )

    @st.fragment(run_every=REFRESH)
    def body() -> None:
        render()

    body()


if __name__ == "__main__":
    main()
