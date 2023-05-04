import logging
from itertools import groupby
from pathlib import Path

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
        raise ValueError("Step must be of the form 10^k")

    if include_stop:
        return [x / factor for x in (list(range(start * factor, stop * factor)) + [stop * factor])]
    return [x / factor for x in range(start * factor, stop * factor)]


def all_equal(iterable):
    g = groupby(iterable)
    return next(g, True) and not next(g, False)