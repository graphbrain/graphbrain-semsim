from functools import lru_cache
from pathlib import Path
from statistics import mean, stdev
import logging

from graphbrain.hypergraph import Hypergraph
from graphbrain_semsim.datasets.config import DATASET_EVAL_DIR
from graphbrain_semsim.datasets.evaluate_dataset import EVALUATION_FILE_SUFFIX
from graphbrain_semsim.datasets.models import DatasetEvaluation, EvaluationResult, StandardDeviation, EvaluationMetrics
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
            f"accuracy={best_evaluation_result.accuracy:.3f}, "
            f"precision={best_evaluation_result.precision:.3f}, "
            f"recall={best_evaluation_result.recall:.3f}, "
            f"f1={best_evaluation_result.f1:.3f}"
            f"mcc={best_evaluation_result.mcc:.3f}, "
        )
        if best_threshold is not None:
            logger.info(f"--> best threshold: {best_threshold:.2f}")
        if best_evaluation.ref_edges is not None:
            logger.info(
                f"--> best ref. edges (n={len(best_evaluation.ref_edges)}, sample mod={best_evaluation.sample_mod}):\n"
                + '\n'.join([hg.text(edge) for edge in best_evaluation.ref_edges]))

    return best_evaluations_results_thresholds


def get_best_evaluation_and_result_and_threshold(
    dataset_evaluations: list[DatasetEvaluation],
    eval_metric: str,
) -> tuple[DatasetEvaluation, EvaluationResult, float | None]:
    # dataset evaluation does not have sub-evaluations (semsim-fix)
    if len(dataset_evaluations) == 1:
        best_result, best_threshold = get_best_result_and_threshold_for_single_eval(
            dataset_evaluations[0].symbolic_eval_result, dataset_evaluations[0].semsim_eval_results, eval_metric
        )
        return dataset_evaluations[0], best_result, best_threshold

    # dataset evaluation has sub-evaluations (semsim-ctx)
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
    add_mean_eval_results: bool = False
) -> dict[str, tuple[EvaluationResult, float | None]]:
    # dictionary mapping (sub) evaluation names to tuples of best results and thresholds
    best_results_and_thresholds: dict[str, tuple[EvaluationResult, float | None]] = {}

    # iterating over all dataset evaluations, which may contain multiple sub-evaluations
    for dataset_eval_name, dataset_sub_evaluations in zip(dataset_eval_names, dataset_evaluations):
        best_sub_results_and_thresholds: list[tuple[EvaluationResult, float | None]] = (
            get_best_results_and_thresholds_for_multi_eval(
                dataset_sub_evaluations, eval_metric, add_mean_eval_results
            )
        )
        # if there are no sub-evaluations, add the best result and threshold to the dictionary
        # by using the dataset evaluation name as the key
        if len(best_sub_results_and_thresholds) == 1:
            best_results_and_thresholds[dataset_eval_name] = best_sub_results_and_thresholds[0]
            continue

        # iterating over all sub-evaluations to generate sub-evaluation names
        # (this is only relevant if there are multiple sub-evaluations)
        for sub_eval_idx, (best_result, best_threshold) in enumerate(best_sub_results_and_thresholds):
            # skip the last sub-evaluation if the flag is set, since it is the mean of all sub-evaluations
            if add_mean_eval_results and sub_eval_idx == len(best_sub_results_and_thresholds) - 1:
                break

            sub_eval: DatasetEvaluation = dataset_sub_evaluations[sub_eval_idx]
            # us the sub-evaluation sample mod as a suffix for the sub-evaluation name
            best_results_and_thresholds[f"{dataset_eval_name}_{sub_eval.sample_mod}"] = best_result, best_threshold

        if add_mean_eval_results:
            # add the mean of all sub-evaluations to the dictionary using the "_mean" suffix for the name
            mean_semsim_best_result, mean_semsim_best_threshold = best_sub_results_and_thresholds[-1]
            mean_semsim_best_result.std_dev = compute_standard_deviation_for_threshold(
                dataset_sub_evaluations[:-1], mean_semsim_best_threshold
            )
            best_results_and_thresholds[f"{dataset_eval_name}_mean"] = (
                mean_semsim_best_result, mean_semsim_best_threshold
            )

            # add the mean of the best results of all sub-evaluations to the dictionary using the "_mean-best" suffix
            best_sub_results: list[EvaluationResult] = [result for result, _ in best_sub_results_and_thresholds[:-1]]
            mean_best_result: EvaluationResult = compute_mean_eval_result(best_sub_results)
            mean_best_result.std_dev = compute_standard_deviation(best_sub_results)
            mean_best_threshold: float = mean([threshold for _, threshold in best_sub_results_and_thresholds[:-1]])
            best_results_and_thresholds[f"{dataset_eval_name}_mean-best"] = mean_best_result, mean_best_threshold

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

    if len(dataset_evaluations) > 1 and append_mean_semsim_eval_results:
        assert all(data_eval.semsim_eval_results for data_eval in dataset_evaluations), (
            "All dataset evaluations in a multi eval must have semsim_eval_results"
        )
        mean_semsim_eval_results: dict[float, EvaluationResult] = compute_mean_semsim_eval_results(dataset_evaluations)
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
        semsim_eval_results: dict[float, EvaluationResult],
        eval_metric: str
) -> tuple[EvaluationResult, float | None]:
    if symbolic_eval_result:
        return symbolic_eval_result, None

    if semsim_eval_results:
        best_threshold, best_result = max(
            semsim_eval_results.items(), key=lambda threshold_result: getattr(threshold_result[1], eval_metric)
        )
        return best_result, best_threshold


def compute_mean_semsim_eval_results(sub_evaluations: list[DatasetEvaluation]) -> dict[float, EvaluationResult]:
    assert all(data_eval.semsim_eval_results for data_eval in sub_evaluations), (
        "All dataset evaluations must have semsim_eval_results"
    )
    # compute mean values for each semsim threshold
    return {
        t: EvaluationResult(**{
            eval_metric: mean(getattr(sub_eval.semsim_eval_results[t], eval_metric) for sub_eval in sub_evaluations)
            for eval_metric in EvaluationMetrics
        }) for t in sub_evaluations[0].semsim_eval_results.keys()
    }


def compute_standard_deviation_for_threshold(
        sub_evaluations: list[DatasetEvaluation], threshold: float
) -> StandardDeviation:
    assert all(data_eval.semsim_eval_results for data_eval in sub_evaluations), (
        "All dataset evaluations must have semsim_eval_results"
    )
    return StandardDeviation(**{
        eval_metric: stdev([sub_eval.semsim_eval_results[threshold][eval_metric] for sub_eval in sub_evaluations])
        for eval_metric in EvaluationMetrics
    })


def compute_mean_eval_result(evaluation_results: list[EvaluationResult]) -> EvaluationResult:
    return EvaluationResult(**{
        eval_metric: mean([evaluation_result[eval_metric] for evaluation_result in evaluation_results])
        for eval_metric in EvaluationMetrics
    })


def compute_standard_deviation(evaluation_results: list[EvaluationResult]) -> StandardDeviation:
    return StandardDeviation(**{
        eval_metric: stdev([evaluation_result[eval_metric] for evaluation_result in evaluation_results])
        for eval_metric in EvaluationMetrics
    })


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


@lru_cache(maxsize=5)
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
