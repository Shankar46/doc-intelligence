"""
Central logging setup. Call setup_logging() once at app startup, then
`logging.getLogger(__name__)` everywhere else — never print().
"""
import logging
import sys

from app.core.config import settings


def setup_logging() -> None:
    level = getattr(logging, settings.log_level.upper(), logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(level)
    # avoid duplicate handlers on reload
    root.handlers = [handler]

    # quiet noisy third-party loggers a bit
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
