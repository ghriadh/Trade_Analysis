import logging
import os
from datetime import datetime


def setup_logger(name: str, log_dir: str = "logs", level: int = logging.INFO) -> logging.Logger:
    """
    Configure a named logger with console and file output.

    The logger is identified by `name`. All child loggers (i.e. modules
    whose __name__ starts with `name + "."`) automatically propagate their
    records to this logger, so a single call to setup_logger("src") in a
    pipeline entrypoint is enough to capture logs from every module under
    the src/ package.

    Args:
        name:    Logger name. Use "src" from pipeline entrypoints so that
                 all src.* module loggers propagate here automatically.
        log_dir: Directory where the timestamped log file is written.
        level:   Logging level (default INFO).

    Returns:
        Configured logging.Logger instance.
    """
    os.makedirs(log_dir, exist_ok=True)

    run_id   = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = os.path.join(log_dir, f"{name}_{run_id}.log")

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False   # prevent records from bubbling to the root logger

    if logger.handlers:
        return logger          # already initialised — avoid duplicate handlers

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    console_handler = logging.StreamHandler()
    console_handler.stream = open(console_handler.stream.fileno(), 'w', encoding='utf-8', closefd=False)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logger.info("Logger initialised | log file: %s", log_file)

    return logger
