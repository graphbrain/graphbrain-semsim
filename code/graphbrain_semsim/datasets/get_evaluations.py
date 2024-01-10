import logging
from functools import lru_cache
from pathlib import Path

from graphbrain_semsim.datasets.config import DATASET_EVAL_DIR
from graphbrain_semsim.datasets.evaluate_dataset import EVALUATION_FILE_SUFFIX
from graphbrain_semsim.datasets.models import DatasetEvaluation
from graphbrain_semsim.utils.file_handling import load_json

logger = logging.getLogger(__name__)


def get_dataset_evaluations(
    dataset_eval_names: list[str],
    case_study: str,
    dataset_id: str,
) -> list[list[DatasetEvaluation]]:
    logger.info(
        f"Getting dataset evaluation data for case study '{case_study}', "
        f"dataset '{dataset_id}' and evaluation names: {dataset_eval_names}..."
    )
    dataset_evaluations: list[list[DatasetEvaluation]] = []
    for dataset_eval_name in dataset_eval_names:
        dataset_evaluations.append(
            get_dataset_evaluations_per_eval_name(
                dataset_eval_name=dataset_eval_name,
                case_study=case_study,
                dataset_id=dataset_id,
            )
        )
    return dataset_evaluations


@lru_cache(maxsize=None)
def get_dataset_evaluations_per_eval_name(
        dataset_eval_name: str,
        case_study: str,
        dataset_id: str,
) -> list[DatasetEvaluation]:
    dataset_eval_id: str = f"{dataset_id}_{EVALUATION_FILE_SUFFIX}_{case_study}_{dataset_eval_name}"

    dataset_evaluation_sub_dir: Path = DATASET_EVAL_DIR / dataset_eval_id
    if dataset_evaluation_sub_dir.is_dir():
        return [
            load_json(sub_evaluation_file_path, DatasetEvaluation, exit_on_error=True)
            for sub_evaluation_file_path in dataset_evaluation_sub_dir.iterdir()
            if sub_evaluation_file_path.suffix == ".json"
        ]

    dataset_evaluation_file_path: Path = DATASET_EVAL_DIR / f"{dataset_eval_id}.json"
    return [load_json(dataset_evaluation_file_path, DatasetEvaluation, exit_on_error=True)]

