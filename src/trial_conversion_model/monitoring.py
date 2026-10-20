import json
import os
from pathlib import Path

import boto3
import pandas as pd
from dotenv import load_dotenv
from evidently import Report
from evidently.presets import DataDriftPreset

from trial_conversion_model.data import RAW_DATA, FEATURES, add_features

REPORT_DIR = Path("monitoring")

# The convention the report is built with: the dataset counts as drifted
# when at least half of the features have drifted individually.
DRIFT_SHARE = 0.5


def load_reference(path: Path = RAW_DATA) -> pd.DataFrame:
    """The training extract is the reference: what normal looked like."""
    return pd.read_csv(path)


def run_drift_check(current: pd.DataFrame, reference: pd.DataFrame) -> object:
    """Compare current trials against the reference, feature by feature.

    Labels for live trials are 11+ days away, so input drift is the
    earliest signal that the world has shifted under the model.
    """
    # Apply the same feature engineering used in training to the current and reference datasets.
    current_features = add_features(current)[Features]
    reference_features = add_features(reference)[Features]

    # Create Evidently report
    report = Report([DataDriftPreset(drift_share=DRIFT_SHARE)])

    # Compare current data and reference data
    result = report.run(
        current_data=current_features,
        reference_data=reference_features
    )

    return result


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
        json.dumps({"name": name, "drifted_share": share, "drift": share >= DRIFT_SHARE})
    )

    bucket = os.environ["S3_BUCKET"]
    s3 = boto3.client("s3")
    for path in (html_path, json_path):
        s3.upload_file(str(path), bucket, f"monitoring/{path.name}")
    return {"drifted_share": share, "drift": share >= DRIFT_SHARE}
