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
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Извлекаем только не-__MACOSX
            for member in zip_ref.namelist():
                if '__MACOSX' not in member:
                    zip_ref.extract(member, target_dir)

    # Рекурсивно удаляем __MACOSX, если она создалась
    for root, dirs, files in os.walk(target_dir, topdown=False):
        if '__MACOSX' in root:
            shutil.rmtree(root)
            print(f"Удалена папка __MACOSX: {root}")  # Дебаггинг

    return target_dir


def find_dicom_dir(unzip_path: str) -> str:
    """
    Находит папку с .dcm файлами внутри unzip_path, включая корень и первую вложенную папку.
    """
    # Проверяем корень
    root_files = os.listdir(unzip_path)
    dcm_files_root = [f for f in root_files if f.lower().endswith('.dcm')]
    if dcm_files_root:
        print(f"Найдена DICOM в корне: {unzip_path} с {len(dcm_files_root)} файлами")  # Дебаггинг
        return unzip_path

    # Проверяем первую вложенную папку (если корень — только папки)
    subdirs = [d for d in root_files if os.path.isdir(os.path.join(unzip_path, d)) and '__MACOSX' not in d]
    if subdirs:
        first_subdir = os.path.join(unzip_path, subdirs[0])
        subdir_files = os.listdir(first_subdir)
        dcm_files_subdir = [f for f in subdir_files if f.lower().endswith('.dcm')]
        if dcm_files_subdir:
            print(f"Найден DICOM в подпапке: {first_subdir} с {len(dcm_files_subdir)} файлами")  # Дебаггинг
            return first_subdir

    # Рекурсивный поиск в случае неудачи (для сложных структур)
    for root, dirs, files in os.walk(unzip_path):
        # Пропускаем служебные папки macOS
        if '__MACOSX' in root:
            continue
        try:
            root_decoded = os.fsdecode(root).decode('utf-8', errors='ignore')
            print(f"Проверяем папку: {root_decoded}")  # Дебаггинг
        except:
            root_decoded = root
            continue  # Пропускаем повреждённые пути

        dcm_files = [f for f in files if f.lower().endswith('.dcm')]
        if dcm_files:
            print(f"Найдена DICOM-папка: {root_decoded} с {len(dcm_files)} файлами")  # Дебаггинг
            return root_decoded

    raise ValueError("Не найдено .dcm файлов в ZIP-архиве.")


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

