"""Lightweight logging helpers (stdout logger + JSONL writer)."""
from __future__ import annotations
import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional


def setup_logger(name: str = "ml4img", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        ch = logging.StreamHandler()
        fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", datefmt="%H:%M:%S")
        ch.setFormatter(fmt)
        logger.addHandler(ch)
    return logger


@dataclass
class JsonlWriter:
    path: str
    auto_mkdir: bool = True

    def __post_init__(self):
        d = os.path.dirname(self.path)
        if self.auto_mkdir and d:
            os.makedirs(d, exist_ok=True)

    def write(self, obj: Dict[str, Any]):
        with open(self.path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
