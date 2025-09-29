import math
import os
import tempfile

import boto3
from botocore.config import Config

import shutil
import zipfile

import numpy as np


def get_s3_client():
    endpoint_url = os.getenv("S3_ENDPOINT")
    region = os.getenv("AWS_REGION", "us-east-1")
    signature_version = os.getenv("S3_SIGNATURE_VERSION", "s3v4")
    config = Config(signature_version=signature_version, s3={"addressing_style": "path"})
    session = boto3.session.Session()
    s3 = session.client(
        "s3",
        endpoint_url=endpoint_url,
        region_name=region,
        aws_access_key_id=os.getenv("S3_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("S3_SECRET_ACCESS_KEY"),
        config=config,
    )
    return s3


def download_zip_from_s3(s3, bucket: str, filename: str) -> str:
    if not filename.endswith(".zip"):
        object_key = f"{filename}.zip"
    else:
        filename = filename[:-4]
        object_key = f"{filename}.zip"

    tmp_dir = tempfile.mkdtemp(prefix="kote_zip_")
    local_zip_path = os.path.join(tmp_dir, f"{filename}.zip")
    s3.download_file(bucket, object_key, local_zip_path)
    return local_zip_path


def unzip_to_temp(zip_path: str) -> str:
    """
    Распаковывает zip в отдельную временную папку и возвращает путь к ней.
    """
    target_dir = tempfile.mkdtemp(prefix="kote_unzip_")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(target_dir)
    return target_dir


def safe_rmtree(path: str):
    try:
        shutil.rmtree(path, ignore_errors=True)
    except Exception:
        pass


def error_to_probability(error_value: float, threshold: float, tau: float = None) -> float:
    """
    Превращает ошибку реконструкции в вероятность патологии [0..1].
    Используем логистическую функцию вокруг threshold.

    tau — "температура" (крутизна). По умолчанию 10% от threshold (но ≥ 1e-6).
    """
    if tau is None:
        tau = max(threshold * 0.1, 1e-6)
    # prob -> 0.5 на threshold; выше threshold -> ближе к 1 (патология)
    prob = 1.0 / (1.0 + math.exp(-(error_value - threshold) / tau))
    # На всякий случай отрежем края
    return float(np.clip(prob, 0.0, 1.0))


def upload_bytes(s3, bucket: str, key: str, data: bytes, content_type: str = "application/octet-stream"):
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=data,
        ContentType=content_type,
        ACL="public-read",
    )

