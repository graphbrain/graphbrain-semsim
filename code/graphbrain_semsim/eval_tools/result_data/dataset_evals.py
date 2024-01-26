from functools import lru_cache
from pathlib import Path
from statistics import mean
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
        best_result, best_threshold = get_best_result_and_threshold_for_single_eval(
            dataset_evaluations[0].symbolic_eval_result, dataset_evaluations[0].semsim_eval_results, eval_metric
        )
        return dataset_evaluations[0], best_result, best_threshold

    results_and_thresholds: list[tuple[EvaluationResult, float]] = [
        get_best_result_and_threshold_for_single_eval(
            dataset_evaluation.symbolic_eval_result, dataset_evaluation.semsim_eval_results, eval_metric
        )
        for dataset_evaluation in dataset_evaluations
    ]
    best_result_idx: int = max(enumerate(results_and_thresholds), key=lambda t: getattr(t[1][0], eval_metric))[0]
    best_evaluation: DatasetEvaluation = dataset_evaluations[best_result_idx]
    best_result, best_threshold = results_and_thresholds[best_result_idx]
    return best_evaluation, best_result, best_threshold


def get_best_results_and_thresholds(
    dataset_evaluations: list[list[DatasetEvaluation]],
    dataset_eval_names: list[str],
    eval_metric: str,
    append_mean_semsim_eval_results: bool = False
) -> dict[str, tuple[EvaluationResult, float | None]]:
    best_results_and_thresholds: dict[str, tuple[EvaluationResult, float | None]] = {}
    for dataset_eval_name, dataset_sub_evaluations in zip(dataset_eval_names, dataset_evaluations):
        best_sub_results_and_thresholds: [tuple[EvaluationResult, float | None]] = (
            get_best_results_and_thresholds_for_multi_eval(
                dataset_sub_evaluations, eval_metric, append_mean_semsim_eval_results
            )
        )
        for sub_eval_idx, (best_result, best_threshold) in enumerate(best_sub_results_and_thresholds):
            sub_eval_name_suffix: str | None = None
            if len(best_sub_results_and_thresholds) > 1:
                sub_eval_name_suffix: str = f"{sub_eval_idx + 1}"

                # last sub-evaluation is the mean of all sub-evaluations
                if append_mean_semsim_eval_results and sub_eval_idx == len(best_sub_results_and_thresholds):
                    sub_eval_name_suffix = "mean"

                    # log standard deviation of the metric for the sub-evaluations
                    scores: list[float] = [
                        getattr(sub_eval_result, eval_metric) for sub_eval_result, _
                        in best_sub_results_and_thresholds
                    ]
                    mean_score: float = mean(scores)
                    standard_deviation: float = mean([(score - mean_score) ** 2 for score in scores]) ** 0.5

                    logger.info(
                        f"Standard deviation of '{eval_metric}' for '{dataset_eval_name}' "
                        f"sub-evaluations: {standard_deviation:.4f}"
                    )

            sub_eval_name: str = f"{dataset_eval_name}" + (f"_{sub_eval_name_suffix}" if sub_eval_name_suffix else "")
            best_results_and_thresholds[sub_eval_name] = best_result, best_threshold

    return best_results_and_thresholds


def get_best_results_and_thresholds_for_multi_eval(
        dataset_evaluations: list[DatasetEvaluation],
        eval_metric: str,
        append_mean_semsim_eval_results: bool = False
) -> list[tuple[EvaluationResult, float | None]]:
    best_results_and_thresholds: list[tuple[EvaluationResult, float | None]] = [
        get_best_result_and_threshold_for_single_eval(
            dataset_evaluation.symbolic_eval_result, dataset_evaluation.semsim_eval_results, eval_metric
        )
        for dataset_evaluation in dataset_evaluations
    ]

    if append_mean_semsim_eval_results and all(data_eval.semsim_eval_results for data_eval in dataset_evaluations):
        mean_semsim_eval_results: dict[float, EvaluationResult] = get_mean_sem_sim_eval_results(dataset_evaluations)
        best_results_and_thresholds.append(
            get_best_result_and_threshold_for_single_eval(
                symbolic_eval_result=None, semsim_eval_results=mean_semsim_eval_results, eval_metric=eval_metric)
        )
        assert len(best_results_and_thresholds) == len(dataset_evaluations) + 1, (
            f"Length of best_results_and_thresholds ({len(best_results_and_thresholds)}) "
            f"does not match length of dataset_evaluations ({len(dataset_evaluations)}) + 1"
        )

    return best_results_and_thresholds


def get_best_result_and_threshold_for_single_eval(
        symbolic_eval_result: EvaluationResult,
        semsim_eval_results: EvaluationResult,
        eval_metric: str
) -> tuple[EvaluationResult, float | None]:
    if symbolic_eval_result:
        return symbolic_eval_result, None

    if semsim_eval_results:
        best_threshold, best_result = max(
            semsim_eval_results.items(), key=lambda threshold_result: getattr(threshold_result[1], eval_metric)
        )
        return best_result, best_threshold


def get_mean_sem_sim_eval_results(sub_evaluations: list[DatasetEvaluation]) -> dict[float, EvaluationResult]:
    # compute mean values for each threshold
    mean_semsim_eval_results: dict[float, EvaluationResult] = {
        t: EvaluationResult(
            accuracy=mean([sub_eval.semsim_eval_results[t].accuracy for sub_eval in sub_evaluations]),
            precision=mean([sub_eval.semsim_eval_results[t].precision for sub_eval in sub_evaluations]),
            recall=mean([sub_eval.semsim_eval_results[t].recall for sub_eval in sub_evaluations]),
            f1=mean([sub_eval.semsim_eval_results[t].f1 for sub_eval in sub_evaluations]),
        )
        for t in sub_evaluations[0].semsim_eval_results.keys()
    }
    return mean_semsim_eval_results


def get_dataset_evaluations(
    dataset_eval_names: list[str],
    case_study: str,
    dataset_id: str,
) -> list[list[DatasetEvaluation]]:
    logger.info(
        f"Getting dataset evaluation data for case study '{case_study}', "
        f"dataset '{dataset_id}' and evaluation names: {dataset_eval_names} ..."
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
