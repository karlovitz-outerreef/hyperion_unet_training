'''
File: s3_utils.py
Desc: Contains functions for writing data to an S3 bucket.
'''

import io
import numpy as np
import pandas as pd
import boto3

s3 = boto3.client("s3")

def save_npy_to_s3(arr: np.ndarray, bucket: str, key: str):
    """
    Save a numpy array directly to S3 as a .npy file.
    """
    if arr.dtype != np.uint8 or arr.ndim != 2:
        raise ValueError(f"Expected 2D uint8 array, got {arr.shape}, {arr.dtype}")

    buf = io.BytesIO()
    np.save(buf, arr, allow_pickle=False)
    buf.seek(0)

    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=buf,
        ContentType="application/octet-stream"
    )


def append_df_to_s3_csv(df: pd.DataFrame, bucket: str, key: str):
    """
    Append a pandas DataFrame to a csv file on s3
    (or create the csv file if it doesn't exist).
    """
    try:
        # Try reading the existing CSV
        obj = s3.get_object(Bucket=bucket, Key=key)
        existing = pd.read_csv(io.BytesIO(obj["Body"].read()))
        combined = pd.concat([existing, df], ignore_index=True)
    except s3.exceptions.NoSuchKey:
        # File doesn't exist yet
        combined = df
    except Exception as e:
        # If it's another error, raise it
        raise

    # Save back to CSV in memory
    buf = io.StringIO()
    combined.to_csv(buf, index=False)
    s3.put_object(Bucket=bucket, Key=key, Body=buf.getvalue(), ContentType="text/csv")