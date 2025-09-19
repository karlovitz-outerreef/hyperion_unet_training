'''
File: mhd_utils.py
Desc: Contains functions for reading a .mhd file from an S3
      bucket and converting it into a numpy array. For
      simplicity, the functionality ASSUMES all .mhd files
      specify 2-dimensional spatial data with 1 channel dimension
      and a data type of MET_UCHAR (since all of Sintef's data
      follows this format).
'''

import boto3
import re
import numpy as np


def load_mhd_u8_from_s3(bucket: str, key_mhd: str):
    """
    Minimal MetaImage loader for MET_UCHAR + 1 channel.
    Returns a NumPy array shaped (H,W) for 2D or (D,H,W) for 3D.
    """
    s3 = boto3.client("s3")

    # --- read header (.mhd as text) ---
    hdr_txt = s3.get_object(Bucket=bucket, Key=key_mhd)["Body"].read().decode("utf-8", "replace")

    def get(field, default=None):
        m = re.search(rf"^\s*{field}\s*=\s*(.+?)\s*$", hdr_txt, flags=re.M)
        return (m.group(1).strip() if m else default)

    # sanity checks
    ndims = int(get("NDims"))
    et = get("ElementType")
    enc = int(get("ElementNumberOfChannels", "1"))
    if ndims != 2 or et.upper() != "MET_UCHAR" or enc != 1:
        raise ValueError(f"Expected MET_UCHAR + 1 channel, got {et}, channels={enc}")
    dimsize = [int(x) for x in get("DimSize").split()]
    if len(dimsize) != ndims:
        raise ValueError(f"DimSize length {len(dimsize)} != NDims {ndims}")

    raw_name = get("ElementDataFile")
    if not raw_name:
        raise ValueError("ElementDataFile missing")
    raw_key = "/".join(key_mhd.split("/")[:-1] + [raw_name])

    # --- read raw bytes and reshape ---
    raw_bytes = s3.get_object(Bucket=bucket, Key=raw_key)["Body"].read()
    arr = np.frombuffer(raw_bytes, dtype=np.uint8)

    expected = np.prod(dimsize)
    if arr.size != expected:
        raise ValueError(f"Raw size mismatch: expected {expected} bytes, got {arr.size}")

    # reshape array into height by width and return
    H, W = dimsize[1], dimsize[0]      # (Y, X)
    return arr.reshape(H, W)