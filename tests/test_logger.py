"""Unit tests for logging utilities."""
import json
import logging
import os
import sqlite3
import tempfile
import uuid
from pathlib import Path
from unittest import mock

import pytest


class TestJSONFormatter:
    def test_output_is_valid_json(self):
        from utils.logger import JSONFormatter

        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="hello world", args=(), exc_info=None,
        )
        output = formatter.format(record)
        data = json.loads(output)
        assert data["message"] == "hello world"
        assert data["level"] == "INFO"
        assert "timestamp" in data

    def test_extra_fields_merged(self):
        from utils.logger import JSONFormatter

        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="msg", args=(), exc_info=None,
        )
        record.extra = {"request_id": "abc-123", "detector": "haar"}  # type: ignore
        data = json.loads(formatter.format(record))
        assert data["request_id"] == "abc-123"
        assert data["detector"] == "haar"


class TestRequestLogger:
    def test_singleton_same_instance(self):
        from utils.logger import RequestLogger

        a = RequestLogger()
        b = RequestLogger()
        assert a is b

    def test_log_request_writes_to_sqlite(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LOG_DIR", str(tmp_path))
        # Reset singleton so it picks up the new LOG_DIR
        import utils.logger as lg
        lg.RequestLogger._instance = None
        # Rebuild LOG_DIR for this test
        import importlib
        importlib.reload(lg)

        logger = lg.RequestLogger()
        rid = str(uuid.uuid4())
        logger.log_request(
            request_id=rid,
            user_id="user-1",
            detector="haar",
            image_metadata={"width": 100, "height": 100, "format": "image/png"},
            face_count=2,
            processing_time_ms=42.5,
        )

        conn = sqlite3.connect(tmp_path / "requests.db")
        row = conn.execute(
            "SELECT * FROM request_logs WHERE request_id = ?", (rid,)
        ).fetchone()
        conn.close()

        assert row is not None
        # request_id is column index 1
        assert row[1] == rid
        # face_count is column 8
        assert row[8] == 2

    def test_log_request_error_status(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LOG_DIR", str(tmp_path))
        import utils.logger as lg
        lg.RequestLogger._instance = None
        import importlib
        importlib.reload(lg)

        logger = lg.RequestLogger()
        rid = str(uuid.uuid4())
        logger.log_request(
            request_id=rid,
            user_id=None,
            detector="yolo",
            image_metadata={"width": 50, "height": 50, "format": "unknown"},
            face_count=0,
            processing_time_ms=10.0,
            status="error",
            error_message="model crashed",
        )

        conn = sqlite3.connect(tmp_path / "requests.db")
        row = conn.execute(
            "SELECT status, error_message FROM request_logs WHERE request_id = ?", (rid,)
        ).fetchone()
        conn.close()

        assert row[0] == "error"
        assert row[1] == "model crashed"
