import io
import boto3
import pandas as pd
from tqdm import tqdm

# -------- config --------
BUCKET = "vessel-segmentation-data"
MANIFEST_KEY = "data/processed/manifest.csv"
FRAME_COL = "frame_path"
SEGM_COL = "segm_path"

# ---- patient-level split consistency check ----
SRC_COL   = "source_dataset"
PID_COL   = "patient_id"
MODEL_COL = "model_dataset"

s3 = boto3.client("s3")


def read_csv_from_s3(bucket: str, key: str) -> pd.DataFrame:
    obj = s3.get_object(Bucket=bucket, Key=key)
    return pd.read_csv(io.BytesIO(obj["Body"].read()))


def parse_s3_uri_or_key(val: str, default_bucket: str):
    """
    Accepts 's3://bucket/key' OR 'key' and returns (bucket, key).
    """
    if pd.isna(val):
        return None
    val = str(val)
    return default_bucket, val.lstrip("/")


def s3_exists(bucket: str, key: str) -> bool:
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except s3.exceptions.ClientError as e:
        if e.response["Error"]["Code"] == "404":
            return False
        return False  # treat any error as "not found"


def check_patient_split_consistency(df: pd.DataFrame,
                                    src_col: str = SRC_COL,
                                    pid_col: str = PID_COL,
                                    model_col: str = MODEL_COL):
    """
    Ensures that for each (source_dataset, patient_id) pair, the model_dataset
    is the same across all rows. Flags groups with >1 unique non-null labels,
    and groups where model_dataset is entirely missing (all null).
    """
    # Work only on the three relevant columns
    sub = df[[src_col, pid_col, model_col]].copy()

    # Normalize labels (strip whitespace, lower) & keep NaN as NaN
    sub[model_col] = sub[model_col].astype("string").str.strip().str.lower()

    # Group by (source_dataset, patient_id)
    g = sub.groupby([src_col, pid_col])[model_col]

    # Unique non-null labels per group
    unique_labels = g.apply(lambda s: sorted({x for x in s.dropna()})).rename("unique_labels")
    counts = g.size().rename("n_rows")
    null_counts = g.apply(lambda s: int(s.isna().sum())).rename("n_nulls")

    report = pd.concat([unique_labels, counts, null_counts], axis=1)

    # Inconsistencies: groups with >1 distinct labels
    inconsistent = report[report["unique_labels"].apply(len) > 1]

    # Missing assignments: groups with 0 labels (all null)
    missing = report[report["unique_labels"].apply(len) == 0]

    return report, inconsistent, missing


# ---- run checks ----
df = read_csv_from_s3(BUCKET, MANIFEST_KEY)

missing_frames = []
missing_segs = []

for idx, row in tqdm(df.iterrows(), total=len(df)):
    # Check frame
    frame = row.get(FRAME_COL)
    if pd.notna(frame):
        bk, ky = parse_s3_uri_or_key(frame, BUCKET)
        if not s3_exists(bk, ky):
            missing_frames.append(frame)

    # Check segmentation
    segm = row.get(SEGM_COL)
    if pd.notna(segm):
        bk, ky = parse_s3_uri_or_key(segm, BUCKET)
        if not s3_exists(bk, ky):
            missing_segs.append(segm)

print("=== Sanity Check Summary ===")
print(f"Frames missing: {len(missing_frames)}")
print(f"Segms  missing: {len(missing_segs)}")

if missing_frames:
    print("\nMissing frame objects (first 20):")
    for x in missing_frames[:20]:
        print(" -", x)

if missing_segs:
    print("\nMissing segmentation objects (first 20):")
    for x in missing_segs[:20]:
        print(" -", x)

# Optional: stop the pipeline if anything is missing
if missing_frames or missing_segs:
    raise RuntimeError("Sanity check failed: some S3 objects are missing.")

report, inconsistent_groups, missing_groups = check_patient_split_consistency(df)

print("\n=== Patient Split Consistency ===")
print(f"Total (source_dataset, patient_id) groups: {len(report)}")
print(f"Inconsistent groups (multiple splits per patient): {len(inconsistent_groups)}")
print(f"Groups with no split assigned (all null): {len(missing_groups)}")

if not inconsistent_groups.empty:
    print("\nExamples of inconsistent groups (up to 10):")
    display_cols = ["unique_labels", "n_rows", "n_nulls"]
    print(inconsistent_groups.head(10)[display_cols].to_string())

if not missing_groups.empty:
    print("\nExamples of groups with missing split (up to 10):")
    display_cols = ["unique_labels", "n_rows", "n_nulls"]
    print(missing_groups.head(10)[display_cols].to_string())

# Fail the check if anything is off
if not inconsistent_groups.empty or not missing_groups.empty:
    raise RuntimeError("Sanity check failed: patient-level split inconsistency or missing split assignments detected.")
