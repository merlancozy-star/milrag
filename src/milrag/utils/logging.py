"""utils/logging.py — 统一日志配置。

CLAUDE.md §9.7: 中文日志消息，英文变量名。
"""
from __future__ import annotations

import logging
import sys


def setup_logging(level: str = "INFO", log_file: str | None = None) -> logging.Logger:
    """配置根日志器：控制台输出 + 可选文件落盘。

    Args:
        level: 日志级别 (DEBUG/INFO/WARNING/ERROR).
        log_file: 若提供，同时写入该文件（实验记录用）.
    Returns:
        配置好的 milrag 根日志器.
    """
    logger = logging.getLogger("milrag")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # 避免重复添加 handler
    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-5s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    logger.addHandler(console)

    if log_file:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


def get_logger(name: str = "milrag") -> logging.Logger:
    """获取子日志器（自动继承根配置）。"""
    return logging.getLogger(name)
