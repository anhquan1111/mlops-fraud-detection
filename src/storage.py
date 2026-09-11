"""Đồng bộ model artifact và báo cáo giám sát với AWS S3."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import boto3
from boto3.exceptions import S3UploadFailedError
from botocore.client import BaseClient
from botocore.exceptions import BotoCoreError, ClientError

logger = logging.getLogger(__name__)

DEFAULT_S3_BUCKET = os.getenv("AWS_S3_BUCKET", "mlops-fraud-detection-artifacts")
AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "ap-southeast-1")


def get_s3_client() -> BaseClient | None:
    """Khởi tạo boto3 S3 client qua biến môi trường hoặc trả về None nếu thiếu cấu hình."""
    try:
        return boto3.client("s3", region_name=AWS_REGION)
    except (BotoCoreError, ClientError) as exc:
        logger.warning(f"Could not initialize S3 client: {exc}")
        return None


def upload_to_s3(
    local_path: Path | str,
    s3_key: str,
    bucket_name: str = DEFAULT_S3_BUCKET,
    client: BaseClient | None = None,
) -> bool:
    """Tải tệp tin cục bộ lên AWS S3 bucket với cơ chế bắt lỗi an toàn."""
    path = Path(local_path)
    if not path.is_file():
        logger.error(f"Cannot upload non-existent file: {path}")
        return False

    s3 = client or get_s3_client()
    if s3 is None:
        logger.warning("S3 client unavailable. Skipping upload.")
        return False

    try:
        logger.info(f"Uploading {path} to s3://{bucket_name}/{s3_key} ...")
        s3.upload_file(str(path), bucket_name, s3_key)
        return True
    except (BotoCoreError, ClientError, S3UploadFailedError, OSError) as exc:
        logger.error(f"S3 upload failed for {path}: {exc}")
        return False


def download_from_s3(
    s3_key: str,
    local_path: Path | str,
    bucket_name: str = DEFAULT_S3_BUCKET,
    client: BaseClient | None = None,
) -> bool:
    """Tải tệp tin từ AWS S3 bucket về hệ thống tệp cục bộ với cơ chế tự tạo thư mục đích."""
    dest = Path(local_path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    s3 = client or get_s3_client()
    if s3 is None:
        logger.warning("S3 client unavailable. Skipping download.")
        return False

    try:
        logger.info(f"Downloading s3://{bucket_name}/{s3_key} to {dest} ...")
        s3.download_file(bucket_name, s3_key, str(dest))
        return True
    except (BotoCoreError, ClientError, OSError) as exc:
        logger.error(f"S3 download failed for {s3_key}: {exc}")
        return False
