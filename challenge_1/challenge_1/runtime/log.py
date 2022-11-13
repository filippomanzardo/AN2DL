import logging
import sys
import warnings
from typing import Any

import colorlog

warnings.filterwarnings("ignore")
_STDOUT = sys.stdout


class StreamToLogger(object):
    """Fake file-like stream object that redirects writes to a logger instance."""

    def __init__(self, logger: logging.Logger, log_level: int) -> None:
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ""

    def write(self, buf: Any) -> None:
        temp_linebuf = self.linebuf + buf
        self.linebuf = ""
        for line in temp_linebuf.splitlines(True):
            # From the io.TextIOWrapper docs:
            #   On output, if newline is None, any '\n' characters written
            #   are translated to the system default line separator.
            # By default sys.stdout.write() expects '\n' newlines and then
            # translates them so this is still cross platform.
            if line[-1] == "\n":
                self.logger.log(self.log_level, line.rstrip())
            else:
                self.linebuf += line

    def flush(self) -> None:
        if self.linebuf != "":
            self.logger.log(self.log_level, self.linebuf.rstrip())
        self.linebuf = ""


def _get_color_formatter(color: str) -> logging.Formatter:
    """Cool colored formatter for log messages."""
    return colorlog.ColoredFormatter(
        fmt=f"[%(asctime)4s]%(fg_thin_{color})s[%(name).8s]]%(reset)s "
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


def restore_stdout(func: Any) -> Any:
    def wrapper(*args: Any, **kwargs: Any) -> None:
        """Restore the stdout stream."""
        current_stdout = sys.stdout
        sys.stdout = _STDOUT
        try:
            func(*args, **kwargs)
        finally:
            print()
        sys.stdout = current_stdout

    return wrapper


def setup(
    log_level: str | int,
) -> None:
    """
    Set up logging for the API app.

    :param log_level: the logging level to use
    """
    root_logger = logging.getLogger()

    handler: logging.Handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_get_color_formatter("red"))
    root_logger.addHandler(handler)
    root_logger.setLevel(log_level)

    stdout_logger = logging.getLogger("STDOUT")
    sys.stdout = StreamToLogger(stdout_logger, logging.INFO)  # type: ignore

    tf_logger = logging.getLogger("tensorflow")
    tf_logger.setLevel(logging.ERROR)
