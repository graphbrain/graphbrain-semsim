import logging
import pickle
from itertools import groupby
from pathlib import Path
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger(__name__)


def save_json(data: BaseModel, result_file_path: Path):
    result_file_path.parent.mkdir(exist_ok=True, parents=True)
    result_file_path.write_text(data.json())
    logger.info(f"Saved to '{result_file_path}'")


def frange(start, stop, step, include_stop: bool = True) -> list[float]:
    factor: float = 1.0 / step
    try:
        factor = int(factor)
    except ValueError:
        raise ValueError("Step must be of the form that 1.0 / step is an integer")

    frange_inlc_stop: list[float] = [
        x / factor for x in (list(range(int(start * factor), int(stop * factor))) + [stop * factor])
    ]

    if include_stop:
        return frange_inlc_stop
        
    return frange_inlc_stop[:-1]


def all_equal(iterable):
    g = groupby(iterable)
    return next(g, True) and not next(g, False)


def save_to_pickle(data: Any, file_path: Path):
    file_path.parent.mkdir(exist_ok=True, parents=True)

    try:
        with file_path.open('wb') as f:
            pickle.dump(data, f)
        logger.info(f"Data successfully saved to {file_path}")
    except Exception as e:
        logger.error(f"Failed to save data to {file_path}. Error: {e}")


def load_from_pickle(file_path: Path):
    if not file_path.exists():
        logger.error(f"File {file_path} does not exist")
        return None

    try:
        with file_path.open('rb') as f:
            data = pickle.load(f)
        logger.info(f"Data successfully loaded from {file_path}")
        return data
    except Exception as e:
        logger.error(f"Failed to load data from {file_path}. Error: {e}")
        return None
