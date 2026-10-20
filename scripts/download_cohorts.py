import boto3
from botocore import UNSIGNED
from botocore.config import Config
from pathlib import Path

BUCKET = "fpds-trial-cohorts"
PREFIX = "cohorts/"
DEST = Path("data/01_raw/cohorts")

DEST.mkdir(parents=True, exist_ok=True)

s3 = boto3.client(
    "s3",
    region_name="eu-north-1",
    config=Config(signature_version=UNSIGNED)
)

response = s3.list_objects_v2(
    Bucket=BUCKET,
    Prefix=PREFIX
)

for obj in response.get("Contents", []):
    key = obj["Key"]

    if key.endswith("/"):
        continue

    destination = DEST / Path(key).name
    s3.download_file(BUCKET, key, str(destination))

    print(f"Downloaded: {destination}")