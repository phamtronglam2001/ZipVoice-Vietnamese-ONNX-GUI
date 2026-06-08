"""
File logging + crash hooks so silent exits leave a trace in logs/.
"""
from __future__ import annotations

import atexit
import faulthandler
import logging
import sys
import traceback
from datetime import datetime
from pathlib import Path

from config import ROOT

LOG_DIR = ROOT / "logs"
LOG_FILE = LOG_DIR / "app.log"
ONNX_LOG_FILE = LOG_DIR / "onnx.log"
CRASH_FILE = LOG_DIR / "crash.log"
ONNX_CRASH_FILE = LOG_DIR / "onnx_crash.log"

_configured = False
_onnx_configured = False


def setup_runtime_logging(
    name: str = "zipvoice_gui",
    log_file: Path | None = None,
    crash_file: Path | None = None,
) -> logging.Logger:
    global _configured, _onnx_configured
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    log_path = log_file or LOG_FILE
    crash_path = crash_file or CRASH_FILE
    is_onnx = log_path.name == "onnx.log"

    logger = logging.getLogger(name)
    if is_onnx:
        if _onnx_configured:
            return logger
    elif _configured:
        return logger

    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if not logger.handlers:
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        logger.addHandler(sh)

    crash_fp = open(crash_path, "a", encoding="utf-8")  # noqa: SIM115
    if not is_onnx:
        faulthandler.enable(crash_fp)

    def _excepthook(exc_type, exc, tb):
        msg = "".join(traceback.format_exception(exc_type, exc, tb))
        logger.critical("Unhandled exception:\n%s", msg)
        crash_fp.write(f"\n--- {datetime.now().isoformat()} ---\n{msg}\n")
        crash_fp.flush()
        sys.__excepthook__(exc_type, exc, tb)

    if not is_onnx:
        sys.excepthook = _excepthook

    @atexit.register
    def _on_exit():
        logger.info("Process exiting.")
        crash_fp.flush()

    if is_onnx:
        _onnx_configured = True
    else:
        _configured = True
    logger.info("Logging started -> %s", log_path)
    return logger


def setup_onnx_logging() -> logging.Logger:
    return setup_runtime_logging(
        name="zipvoice_onnx",
        log_file=ONNX_LOG_FILE,
        crash_file=ONNX_CRASH_FILE,
    )
