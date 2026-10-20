from pathlib import Path
import boto3
from botocore import UNSIGNED
from botocore.config import Config

BUCKET = "fpds-trial-cohorts"
PREFIX = "cohorts/"

s3 = boto3.client(
    "s3",
    region_name="eu-north-1",
    config=Config(signature_version=UNSIGNED),
)

# Retrieve all Parquet files
paginator = s3.get_paginator("list_objects_v2")
keys = []

for page in paginator.paginate(Bucket=BUCKET, Prefix=PREFIX):
    for obj in page.get("Contents", []):
        key = obj["Key"]
        if key.endswith(".parquet"):
            keys.append(key)

# Assumes ISO-formatted dates (YYYY-MM-DD) in filenames
keys = sorted(keys)

if not keys:
    raise RuntimeError("No Parquet cohorts found")

oldest = keys[0]
newest = keys[-1]

print(f"Oldest: {oldest}")
print(f"Newest: {newest}")

output = Path("data/cohorts")
output.mkdir(parents=True, exist_ok=True)

for name, key in [("newest", newest), ("oldest", oldest)]:
    destination = output / f"{name}.parquet"
    s3.download_file(BUCKET, key, str(destination))
    print(f"Downloaded: {destination}")