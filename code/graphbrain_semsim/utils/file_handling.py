import json
import logging
import pickle
from pathlib import Path
from typing import Type, Any, TypeVar

from pydantic import BaseModel

from graphbrain_semsim import parse_config

logger = logging.getLogger(__name__)

BaseModelType = TypeVar('BaseModelType', bound=BaseModel)


def save_json(data: BaseModel, file_path: Path):
    file_path.parent.mkdir(exist_ok=True, parents=True)
    file_path.write_text(data.model_dump_json(), encoding="utf-8")
    logger.info(f"Saved to '{file_path}'")


def load_json(
        file_path: Path,
        model: Type[BaseModelType],
        exit_on_error: bool = False,
        skip_validation: bool = parse_config.SKIP_VALIDATION
) -> BaseModelType | None:
    if not file_path.exists():
        error_msg = f"File {file_path} does not exist"
        if exit_on_error:
            raise ValueError(error_msg)
        logger.error(error_msg)
        return None

    try:
        model_dict: dict = json.loads(file_path.read_text(encoding="utf-8"))
        if skip_validation:
            return model.model_construct(**model_dict)
        return model.model_validate(json.loads(file_path.read_text(encoding="utf-8")))
    except Exception as e:
        logger.error(f"Failed to load data from {file_path}. {e.__class__.__name__}: {e}")
        if exit_on_error:
            exit(1)
        return None


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
