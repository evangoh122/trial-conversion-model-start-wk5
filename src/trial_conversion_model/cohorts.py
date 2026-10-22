from pathlib import Path

import boto3
from botocore import UNSIGNED
from botocore.config import Config

# The course-side pipeline publishes one dated day-3 cohort file per day.
# The bucket is public and read-only: pulling from it needs no credentials
# and involves no coordination between AWS accounts.
COHORT_BUCKET = "fpds-trial-cohorts"
COHORT_PREFIX = "cohorts/"
COHORT_DIR = Path("data/01_raw/cohorts")


def pull_newest_cohort() -> Path:
    """Download the newest day-3 cohort file and return its local path.

    Nothing depends on pulling every day: a catch-up run after missed
    days simply gets the most recent file.
    """
    s3 = boto3.client(
        "s3", region_name="eu-north-1", config=Config(signature_version=UNSIGNED)
    )
    listing = s3.list_objects_v2(Bucket=COHORT_BUCKET, Prefix=COHORT_PREFIX)
    # Names are cohort_YYYY-MM-DD.parquet. ISO dates sort lexicographically,
    # so the alphabetically last key is also the chronologically newest one;
    # no date parsing and no reliance on the listing's own order.
    keys = [
        obj["Key"]
        for obj in listing.get("Contents", [])
        if obj["Key"].endswith(".parquet")
    ]
    if not keys:
        raise FileNotFoundError(
            f"no cohort files under s3://{COHORT_BUCKET}/{COHORT_PREFIX}"
        )
    key = max(keys)
    COHORT_DIR.mkdir(parents=True, exist_ok=True)
    path = COHORT_DIR / Path(key).name
    s3.download_file(COHORT_BUCKET, key, str(path))
    return path


def newest_local_cohort() -> Path:
    """The most recent cohort already pulled to disk."""
    files = sorted(COHORT_DIR.glob("cohort_*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"no cohort files in {COHORT_DIR}; run scripts/score.py first"
        )
    return files[-1]
