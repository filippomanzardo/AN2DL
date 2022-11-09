import logging
import sys

import colorlog


def _get_color_formatter(entity: str, color: str) -> logging.Formatter:
    """Cool colored formatter for log messages."""

    return colorlog.ColoredFormatter(
        fmt=f"[%(asctime)4s]%(fg_thin_{color})s[{entity}]%(reset)s "
        f"%(log_color)s%(levelname)8s |%(reset)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        log_colors={
            "DEBUG": "green",
            "INFO": "blue",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bold_red",
        },
    )


def setup(
    log_level: str | int,
) -> None:
    """
    Set up logging for the API app.

    :param log_level: the logging level to use
    """

    root_logger = logging.getLogger()

    handler: logging.Handler = colorlog.StreamHandler(sys.stdout)
    handler.setFormatter(_get_color_formatter("System", "red"))
    root_logger.addHandler(handler)
    root_logger.setLevel(log_level)
