from functools import lru_cache
from pathlib import Path
import logging

from graphbrain.hypergraph import Hypergraph
from graphbrain_semsim.datasets.config import DATASET_EVAL_DIR
from graphbrain_semsim.datasets.evaluate_dataset import EVALUATION_FILE_SUFFIX
from graphbrain_semsim.datasets.models import DatasetEvaluation, EvaluationResult
from graphbrain_semsim.utils.file_handling import load_json


logger = logging.getLogger(__name__)


def get_best_evaluations_and_results_and_thresholds(
    dataset_evaluations: list[list[DatasetEvaluation]],
    dataset_eval_names: list[str],
    eval_metric: str,
    hg: Hypergraph
) -> dict[str, tuple[DatasetEvaluation, EvaluationResult, float | None]]:
    best_evaluations_results_thresholds: dict[str, tuple[DatasetEvaluation, EvaluationResult, float | None]] = {}
    for dataset_eval_name, dataset_sub_evaluations in zip(dataset_eval_names, dataset_evaluations):
        best_evaluation, best_evaluation_result, best_threshold = get_best_evaluation_and_result_and_threshold(
            dataset_sub_evaluations, eval_metric
        )
        best_evaluations_results_thresholds[dataset_eval_name] = best_evaluation, best_evaluation_result, best_threshold

        logger.info(f"Best evaluation result for dataset evaluation {dataset_eval_name}:")
        logger.info(
            f"--> evaluation metrics ({eval_metric} used for ranking):\n"
            f"accuracy={best_evaluation_result.accuracy:.2f}, "
            f"precision={best_evaluation_result.precision:.2f}, "
            f"recall={best_evaluation_result.recall:.2f}, "
            f"f1={best_evaluation_result.f1:.2f}"
        )
        if best_threshold is not None:
            logger.info(f"--> best threshold: {best_threshold}")
        if best_evaluation.ref_edges is not None:
            logger.info("--> best ref. edges:\n" + '\n'.join([hg.text(edge) for edge in best_evaluation.ref_edges]))

    return best_evaluations_results_thresholds


def get_best_evaluation_and_result_and_threshold(
    dataset_evaluations: list[DatasetEvaluation],
    eval_metric: str,
) -> tuple[DatasetEvaluation, EvaluationResult, float | None]:
    if len(dataset_evaluations) == 1:
        best_result, best_threshold = get_best_result_and_threshold(dataset_evaluations[0], eval_metric)
        return dataset_evaluations[0], best_result, best_threshold

    results_and_thresholds: list[tuple[EvaluationResult, float]] = [
        get_best_result_and_threshold(dataset_evaluation, eval_metric)
        for dataset_evaluation in dataset_evaluations
    ]
    best_result_idx: int = max(enumerate(results_and_thresholds), key=lambda t: getattr(t[1][0], eval_metric))[0]
    best_evaluation: DatasetEvaluation = dataset_evaluations[best_result_idx]
    best_result, best_threshold = results_and_thresholds[best_result_idx]
    return best_evaluation, best_result, best_threshold


def get_best_result_and_threshold(
    dataset_evaluation: DatasetEvaluation,
    eval_metric: str,
) -> tuple[EvaluationResult, float | None]:
    if dataset_evaluation.symbolic_eval_result:
        return dataset_evaluation.symbolic_eval_results, None

    if dataset_evaluation.semsim_eval_results:
        best_threshold, best_result = max(
            dataset_evaluation.semsim_eval_results.items(),
            key=lambda threshold_result: getattr(threshold_result[1], eval_metric)
        )
        return best_result, best_threshold


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
