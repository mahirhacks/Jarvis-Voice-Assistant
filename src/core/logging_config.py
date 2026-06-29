"""Logging configuration."""

import logging


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    for noisy in (
        "comtypes", "httpcore", "httpx", "urllib3", "filelock",
        "httpcore.http11", "faster_whisper", "ctranslate2", "yt_dlp",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)
