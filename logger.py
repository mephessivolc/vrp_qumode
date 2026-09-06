import logging
from pathlib import Path
from typing import Union


def setup_logger(
    name: str = "CVRP_Logger",
    log_file: Union[str, Path] = "execution.log",
    level: int = logging.INFO,
) -> logging.Logger:
    """Configura e retorna um logger que grava no console e em arquivo."""
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Limpa handlers anteriores para evitar duplicidade de mensagens
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 1. Handler para salvar em arquivo
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 2. Handler para exibir no terminal
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger