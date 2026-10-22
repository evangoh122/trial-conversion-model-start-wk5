import json
import logging
import os
from pathlib import Path

import boto3
import pandas as pd
import requests
from botocore.exceptions import BotoCoreError
from dotenv import load_dotenv
from evidently import Report
from evidently.presets import DataDriftPreset

from trial_conversion_model.data import RAW_DATA
from trial_conversion_model.features import FEATURES, add_features

logger = logging.getLogger(__name__)

REPORT_DIR = Path("monitoring")
# The reference sits in the bucket next to models/ and predictions/.
REFERENCE_KEY = "data/trial_snapshot.csv"

# The convention the report is built with: the dataset counts as drifted
# when at least half of the features have drifted individually.
DRIFT_SHARE = 0.5

# A webhook that hangs must not hold up the job that just checked for drift.
SLACK_TIMEOUT = 10


def load_reference(path: Path = RAW_DATA) -> pd.DataFrame:
    """The training extract is the reference: what normal looked like.

    The jobs image carries no data, so the first run on a fresh machine
    fetches the extract from the bucket where it was published once;
    later runs find it on disk.
    """
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        s3 = boto3.client("s3")
        s3.download_file(os.environ["S3_BUCKET"], REFERENCE_KEY, str(path))
    return pd.read_csv(path)


def run_drift_check(current: pd.DataFrame, reference: pd.DataFrame) -> object:
    """Compare current trials against the reference, feature by feature.

    Labels for live trials are 11+ days away, so input drift is the
    earliest signal that the world has shifted under the model.
    """
    ref = add_features(reference)[FEATURES]
    cur = add_features(current)[FEATURES]
    report = Report([DataDriftPreset(drift_share=DRIFT_SHARE)])
    return report.run(reference_data=ref, current_data=cur)


def drifted_share(snapshot: object) -> float:
    """Pull the share of drifted feature columns out of the report."""
    for metric in json.loads(snapshot.json())["metrics"]:
        if metric["metric_name"].startswith("DriftedColumnsCount"):
            return float(metric["value"]["share"])
    raise ValueError("drift metric missing from report")


def publish_report(snapshot: object, name: str) -> dict:
    """Write the report locally (HTML for eyes, JSON for machines) and to S3."""
    load_dotenv()
    REPORT_DIR.mkdir(exist_ok=True)
    html_path = REPORT_DIR / f"{name}.html"
    json_path = REPORT_DIR / f"{name}.json"
    snapshot.save_html(str(html_path))
    share = drifted_share(snapshot)
    json_path.write_text(
        json.dumps(
            {"name": name, "drifted_share": share, "drift": share >= DRIFT_SHARE}
        )
    )

    bucket = os.environ["S3_BUCKET"]
    s3 = boto3.client("s3")
    for path in (html_path, json_path):
        s3.upload_file(str(path), bucket, f"monitoring/{path.name}")
    return {"drifted_share": share, "drift": share >= DRIFT_SHARE}


def report_link(name: str) -> str | None:
    """A week-long link to the HTML report, so the alert is one tap away."""
    bucket = os.environ.get("S3_BUCKET")
    if not bucket:
        return None
    try:
        return boto3.client("s3").generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": f"monitoring/{name}.html"},
            ExpiresIn=7 * 24 * 3600,
        )
    except BotoCoreError:
        return None


def send_alert(result: dict, name: str) -> bool:
    """Post a drift verdict to Slack, if a webhook is configured.

    Alerting is best effort on purpose: a missing URL or a Slack outage must
    never stop the drift job that produced the verdict. Returns whether the
    message went out.
    """
    load_dotenv()
    url = os.environ.get("SLACK_WEBHOOK_URL")
    if not url:
        return False

    verdict = "Drift detected" if result["drift"] else "No drift"
    link = report_link(name)
    report = (
        f"<{link}|Open the Evidently report>"
        if link
        else f"Report: `s3://{os.environ.get('S3_BUCKET', '')}/monitoring/{name}.html`"
    )
    text = (
        f"*{verdict}* in `{name}` — {result['drifted_share']:.0%} of features "
        f"drifted (threshold {DRIFT_SHARE:.0%}).\n{report}"
    )
    try:
        response = requests.post(url, json={"text": text}, timeout=SLACK_TIMEOUT)
        response.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("slack alert failed: %s", exc)
        return False
    return True
