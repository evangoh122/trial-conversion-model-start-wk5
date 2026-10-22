import os
import sys
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

from trial_conversion_model.monitoring import (
    load_reference,
    publish_report,
    run_drift_check,
)


def alert(message: str) -> None:
    webhook = os.environ.get("SLACK_WEBHOOK_URL")

    if not webhook:
        print("SLACK_WEBHOOK_URL not set; alert printed only")
        print(message)
        return

    response = requests.post(
        webhook,
        json={"text": message},
        timeout=10,
    )
    response.raise_for_status()
    print("Slack alert sent successfully")


if __name__ == "__main__":
    # Optional cohort file argument
    if len(sys.argv) > 2:
        sys.exit(
            "usage: uv run scripts/check_drift.py [cohort file]"
        )

    if len(sys.argv) == 2:
        cohort_path = Path(sys.argv[1])
    else:
        # Automatically select the newest cohort
        cohort_dir = Path("data/cohorts")
        cohort_files = list(cohort_dir.glob("*.parquet"))

        if not cohort_files:
            sys.exit(f"No cohort files found in {cohort_dir}")

        cohort_path = max(
            cohort_files,
            key=lambda p: p.stat().st_mtime,
        )

    print(f"Checking cohort: {cohort_path}")

    # Read current cohort
    current = pd.read_parquet(cohort_path)

    # Load reference data
    reference = load_reference()

    # Check for drift
    snapshot = run_drift_check(current, reference)

    # Publish the report
    result = publish_report(snapshot, cohort_path.stem)

    verdict = "DRIFT" if result["drift"] else "OK"

    message = (
        f"{verdict}: {result['drifted_share']:.0%} "
        f"of features drifted ({cohort_path.name})"
    )

    print(message)

    # Only notify Slack when drift is detected
    if result["drift"]:
        alert(message)