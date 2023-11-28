import json
import logging
import pickle
from itertools import groupby
from pathlib import Path
from typing import Any, Type, TypeVar

from pydantic import BaseModel

logger = logging.getLogger(__name__)

BaseModelType = TypeVar('BaseModelType', bound=BaseModel)


def save_json(data: BaseModel, file_path: Path):
    file_path.parent.mkdir(exist_ok=True, parents=True)
    file_path.write_text(data.model_dump_json(), encoding="utf-8")
    logger.info(f"Saved to '{file_path}'")


def load_json(file_path: Path, model: Type[BaseModelType], exit_on_error: bool = False) -> BaseModelType | None:
    if not file_path.exists():
        error_msg = f"File {file_path} does not exist"
        if exit_on_error:
            raise ValueError(error_msg)
        logger.error(error_msg)
        return None

    try:
        return model.model_validate(json.loads(file_path.read_text(encoding="utf-8")))
    except Exception as e:
        logger.error(f"Failed to load data from {file_path}. {e.__class__.__name__}: {e}")
        if exit_on_error:
            exit(1)
        return None


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
