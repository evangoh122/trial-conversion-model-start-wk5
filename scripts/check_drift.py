import sys
from pathlib import Path

import pandas as pd

from trial_conversion_model.monitoring import (
    load_reference,
    publish_report,
    run_drift_check,
    send_alert,
)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: uv run scripts/check_drift.py <cohort file>")
    cohort_path = Path(sys.argv[1])

    # Read current cohort
    current = pd.read_parquet(cohort_path)

    # Load reference data
    reference = load_reference()

    # Check for drift
    snapshot = run_drift_check(current, reference)

    # Publish the report
    result = publish_report(snapshot, cohort_path.stem)

    verdict = "DRIFT" if result["drift"] else "OK"
    print(
        f"{verdict}: {result['drifted_share']:.0%} of features drifted ({cohort_path.name})"
    )

    # Only drift is news; a clean check stays quiet so the channel means
    # something when it does speak.
    if result["drift"]:
        alerted = send_alert(result, cohort_path.stem)
        print(
            "alert sent to Slack"
            if alerted
            else "alert not sent (webhook unset or Slack unreachable)"
        )
