"""Structured logging for oa-bench.

Two outputs:
  - Console (rich, human-readable)
  - Rotating file handler (JSON lines, machine-readable)

Configure once at CLI startup; all modules use `logging.getLogger("oa_bench.*")`.
"""

from __future__ import annotations
import json
import logging
import logging.handlers
import os
import sys
from pathlib import Path


class JSONFormatter(logging.Formatter):
    """Emit each log record as a single JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, datefmt="%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        # Include user-supplied extras
        for k, v in record.__dict__.items():
            if k.startswith("_") or k in (
                "args", "asctime", "created", "exc_info", "exc_text",
                "filename", "funcName", "levelname", "levelno", "lineno",
                "message", "module", "msecs", "msg", "name", "pathname",
                "process", "processName", "relativeCreated", "stack_info",
                "thread", "threadName",
            ):
                continue
            try:
                json.dumps(v)
                payload[k] = v
            except (TypeError, ValueError):
                payload[k] = repr(v)
        return json.dumps(payload, default=str)


def configure_logging(
    output_dir: Path | None = None,
    level: str = "INFO",
    console: bool = True,
    json_file: bool = True,
) -> None:
    """Configure root oa_bench logger.

    Args:
      output_dir: if set, write JSON lines to {output_dir}/oa_bench.log
      level: logging level (DEBUG / INFO / WARNING / ERROR)
      console: emit human-readable to stderr
      json_file: write JSON lines log file (rotates at 10MB, 5 backups)
    """
    lvl = getattr(logging, level.upper(), logging.INFO)
    root = logging.getLogger("oa_bench")
    root.setLevel(lvl)
    # Remove existing handlers (idempotent reconfiguration)
    for h in list(root.handlers):
        root.removeHandler(h)

    if console:
        sh = logging.StreamHandler(sys.stderr)
        sh.setLevel(lvl)
        sh.setFormatter(logging.Formatter(
            "%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
            datefmt="%H:%M:%S",
        ))
        root.addHandler(sh)

    if json_file and output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        fh = logging.handlers.RotatingFileHandler(
            Path(output_dir) / "oa_bench.log",
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        fh.setLevel(lvl)
        fh.setFormatter(JSONFormatter())
        root.addHandler(fh)

    # Silence noisy 3rd-party loggers
    for noisy in ("httpx", "httpcore", "anthropic", "openai", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_log_level_from_env() -> str:
    return os.environ.get("OA_LOG_LEVEL", "INFO").upper()
