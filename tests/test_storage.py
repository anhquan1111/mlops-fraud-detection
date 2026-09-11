"""Unit tests for AWS S3 storage synchronization (src/storage.py)."""

from __future__ import annotations

from unittest.mock import MagicMock

from botocore.exceptions import ClientError

from src.storage import download_from_s3, upload_to_s3


class TestStorageModule:
    """Test suite for S3 upload and download operations with mocks."""

    def test_upload_to_s3_success(self, tmp_path) -> None:
        dummy_file = tmp_path / "test.txt"
        dummy_file.write_text("hello s3")

        mock_s3 = MagicMock()
        success = upload_to_s3(dummy_file, "reports/test.txt", "my-bucket", client=mock_s3)
        assert success is True
        mock_s3.upload_file.assert_called_once_with(
            str(dummy_file), "my-bucket", "reports/test.txt"
        )

    def test_upload_non_existent_file_returns_false(self, tmp_path) -> None:
        missing_file = tmp_path / "missing.txt"
        mock_s3 = MagicMock()
        success = upload_to_s3(missing_file, "reports/missing.txt", "my-bucket", client=mock_s3)
        assert success is False
        mock_s3.upload_file.assert_not_called()

    def test_upload_handles_client_error(self, tmp_path) -> None:
        dummy_file = tmp_path / "test.txt"
        dummy_file.write_text("hello s3")

        mock_s3 = MagicMock()
        mock_s3.upload_file.side_effect = ClientError(
            {"Error": {"Code": "403", "Message": "Forbidden"}}, "upload_file"
        )

        success = upload_to_s3(dummy_file, "reports/test.txt", "my-bucket", client=mock_s3)
        assert success is False

    def test_download_from_s3_success(self, tmp_path) -> None:
        dest_file = tmp_path / "downloaded.txt"
        mock_s3 = MagicMock()

        success = download_from_s3("reports/downloaded.txt", dest_file, "my-bucket", client=mock_s3)
        assert success is True
        mock_s3.download_file.assert_called_once_with(
            "my-bucket", "reports/downloaded.txt", str(dest_file)
        )
