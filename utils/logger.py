from __future__ import annotations

import json
import logging
import logging.handlers
import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


# Default location for logs/. Can be overridden by setting LOG_DIR=/some/path
# in the environment — handy inside Docker.
LOG_DIR = Path(os.environ.get("LOG_DIR", "logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)


# Each log line ends up as a single JSON object. Makes it easy to grep
# and to pipe into tools like jq or a log aggregator later.
class JSONFormatter(logging.Formatter):

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # If the caller attached an "extra" dict (we do this inside
        # RequestLogger), merge it in so those fields become top-level.
        extra = getattr(record, "extra", None)
        if isinstance(extra, dict):
            payload.update(extra)
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def setup_logger(name: str = "face_detection") -> logging.Logger:
    # Returns a logger with JSON output to both the console and a rotating file.
    # Called multiple times during startup — the handler check prevents duplicates.
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    formatter = JSONFormatter()

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    # Rotate at 10 MB, keep the last 5 files. Tweak if disk usage gets crazy.
    file_handler = logging.handlers.RotatingFileHandler(
        LOG_DIR / "app.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Don't pass records up to the root logger (avoids duplicate console output).
    logger.propagate = False
    return logger


# One RequestLogger per process (singleton). Keeps a single SQLite file
# handle and a single lock — simpler than passing it around everywhere.
class RequestLogger:

    _instance: Optional[RequestLogger] = None
    _class_lock = threading.Lock()

    def __new__(cls) -> RequestLogger:
        # Classic double-checked locking so concurrent requests at startup
        # don't create two instances.
        with cls._class_lock:
            if cls._instance is None:
                inst = super().__new__(cls)
                inst._initialized = False
                cls._instance = inst
        return cls._instance

    def __init__(self) -> None:
        # __init__ runs every time you call RequestLogger(), but we only
        # want to set things up once.
        if self._initialized:
            return
        self.db_path = LOG_DIR / "requests.db"
        self.db_lock = threading.Lock()
        self.logger = setup_logger("face_detection.requests")
        self._init_db()
        self._initialized = True

    def _init_db(self) -> None:
        # Create the table on first run. Safe to call repeatedly.
        with self.db_lock:
            conn = sqlite3.connect(self.db_path)
            try:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS request_logs (
                        id               INTEGER PRIMARY KEY AUTOINCREMENT,
                        request_id       TEXT    NOT NULL,
                        user_id          TEXT,
                        timestamp        TEXT    NOT NULL,
                        detector         TEXT    NOT NULL,
                        image_width      INTEGER,
                        image_height     INTEGER,
                        image_format     TEXT,
                        face_count       INTEGER,
                        processing_time_ms REAL,
                        status           TEXT    NOT NULL DEFAULT 'success',
                        error_message    TEXT
                    )
                    """
                )
                conn.commit()
            finally:
                conn.close()

    def log_request(
        self,
        *,
        request_id: str,
        user_id: Optional[str],
        detector: str,
        image_metadata: Dict[str, Any],
        face_count: int,
        processing_time_ms: float,
        status: str = "success",
        error_message: Optional[str] = None,
    ) -> None:
        # Write a single request to the JSON log file AND the SQLite DB.
        # File log is for humans / log shipping; SQLite is for the /ui/logs
        # page and ad-hoc queries.
        extra: Dict[str, Any] = {
            "request_id": request_id,
            "user_id": user_id,
            "detector": detector,
            "image_metadata": image_metadata,
            "face_count": face_count,
            "processing_time_ms": round(processing_time_ms, 3),
            "status": status,
        }
        if error_message:
            extra["error"] = error_message

        # Build a log record manually so we can attach the extra dict
        # without polluting the message string.
        record = self.logger.makeRecord(
            self.logger.name,
            logging.INFO if status == "success" else logging.ERROR,
            fn="",
            lno=0,
            msg="request_processed",
            args=(),
            exc_info=None,
        )
        record.extra = extra  # type: ignore[attr-defined]
        self.logger.handle(record)

        # SQLite write — serialised with the lock so multiple workers in
        # the threadpool don't stomp on each other.
        now = datetime.now(tz=timezone.utc).isoformat()
        with self.db_lock:
            conn = sqlite3.connect(self.db_path)
            try:
                conn.execute(
                    """
                    INSERT INTO request_logs
                        (request_id, user_id, timestamp, detector,
                         image_width, image_height, image_format,
                         face_count, processing_time_ms, status, error_message)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        request_id,
                        user_id,
                        now,
                        detector,
                        image_metadata.get("width"),
                        image_metadata.get("height"),
                        image_metadata.get("format", "unknown"),
                        face_count,
                        round(processing_time_ms, 3),
                        status,
                        error_message,
                    ),
                )
                conn.commit()
            finally:
                conn.close()
