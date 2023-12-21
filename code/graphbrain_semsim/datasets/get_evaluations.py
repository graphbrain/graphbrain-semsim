from pathlib import Path

from graphbrain_semsim.datasets.config import DATASET_EVAL_DIR
from graphbrain_semsim.datasets.evaluate_dataset import EVALUATION_FILE_SUFFIX
from graphbrain_semsim.datasets.models import DatasetEvaluation
from graphbrain_semsim.utils.file_handling import load_json


def get_dataset_evaluations(
    dataset_eval_names: list[str],
    case_study: str,
    dataset_id: str,
) -> dict[int, list[DatasetEvaluation]]:
    dataset_evaluations: dict[int, list[DatasetEvaluation]] = {}
    for dataset_eval_idx, dataset_eval_name in enumerate(dataset_eval_names):
        dataset_eval_id: str = f"{dataset_id}_{EVALUATION_FILE_SUFFIX}_{case_study}_{dataset_eval_name}"

        dataset_evaluation_sub_dir: Path = DATASET_EVAL_DIR / dataset_eval_id
        if dataset_evaluation_sub_dir.is_dir():
            dataset_evaluations[dataset_eval_idx] = [
                load_json(sub_evaluation_file_path, DatasetEvaluation, exit_on_error=True)
                for sub_evaluation_file_path in dataset_evaluation_sub_dir.iterdir()
                if sub_evaluation_file_path.suffix == ".json"
            ]

        else:
            dataset_evaluation_file_path: Path = DATASET_EVAL_DIR / f"{dataset_eval_id}.json"
            dataset_evaluations[dataset_eval_idx] = [load_json(
                dataset_evaluation_file_path, DatasetEvaluation, exit_on_error=True
            )]
    return dataset_evaluations
