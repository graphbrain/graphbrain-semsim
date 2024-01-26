import sys
from pathlib import Path
import logging

import pandas as pd
# from pympler import asizeof

from graphbrain_semsim.datasets.models import DatasetEvaluation, EvaluationResult
from graphbrain_semsim.eval_tools.result_data.dataset_evals import (
    get_dataset_evaluations, get_best_results_and_thresholds
)

from graphbrain_semsim.case_studies.conflicts.config import CASE_STUDY
from graphbrain_semsim.utils.general import all_equal

# from graphbrain_semsim import config
# config.SKIP_VALIDATION = True

logger = logging.getLogger(__name__)


PATTERN_NAME_REPLACEMENTS = {
    "1-1_original-pattern": "original",
    "2-1_pred_semsim-fix_wildcard": "semsim-fix",
    "2-2_pred_semsim-fix-lemma_wildcard": "semsim-fix-lemma",
    "2-3_pred_semsim-ctx_wildcard": "semsim-ctx",
}


def prettify_eval_name(eval_name: str) -> str:
    for pattern_name, short_pattern_name in PATTERN_NAME_REPLACEMENTS.items():
        if pattern_name in eval_name:
            eval_name = eval_name.replace(pattern_name, short_pattern_name)

    if "semsim-ctx" in eval_name:
        last_underscore_idx = eval_name.rfind("_")
        if last_underscore_idx != -1:
            eval_name = eval_name[:last_underscore_idx] + "-" + eval_name[last_underscore_idx + 1:]

    eval_name = eval_name.replace("nref-", "r-")
    eval_name = eval_name.replace("_", " ")

    return eval_name


def insert_latex_column_breaks(eval_name: str) -> str:
    eval_name = eval_name.replace(" ", " & ")

    required_num_breaks = 2
    num_breaks = eval_name.count("&")
    eval_name += " & " * (required_num_breaks - num_breaks)

    return eval_name


def make_results_table(
    dataset_name: str,
    dataset_eval_names: list[str],
    case_study: str = CASE_STUDY,
):
    dataset_id: str = f"dataset_{case_study}_{dataset_name}"

    dataset_evaluations: list[list[DatasetEvaluation]] = get_dataset_evaluations(
        dataset_eval_names, case_study, dataset_id
    )

    logger.info(f"Loaded dataset evaluations. Making results table...")

    # DEBUGGING
    # for dataset_eval_idx, dataset_sub_evaluations in enumerate(dataset_evaluations):
    #     for sub_idx, dataset_sub_evaluation in enumerate(dataset_sub_evaluations):
    #         eval_size = asizeof.asizeof(dataset_sub_evaluation)
    #         print(
    #             f"{dataset_eval_names[dataset_eval_idx]} {sub_idx}: "
    #             f"{eval_size} B, {eval_size / 1024} KB, {eval_size / 1024 / 1024} MB"
    #         )

    metrics: list[str] = ['precision', 'recall', 'f1']

    # Load best evaluations and results for each metric
    # best_results_and_thresholds_per_metric: dict[str, dict[str, tuple[EvaluationResult, float | None]]] = {
    #     metric: get_best_results_and_thresholds(dataset_evaluations, dataset_eval_names, metric)
    #     for metric in metrics
    # }

    # Reorganize the data to have the evaluation names as keys
    # assert all_equal(best_results_and_thresholds_per_metric[metric].keys() for metric in metrics)
    # best_scores_and_thresholds_per_eval_name: dict[str, dict[str, tuple[float, float | None]]] = {
    #     eval_name: {
    #         metric: (
    #             getattr(best_results_and_thresholds_per_metric[metric][eval_name][0], metric),
    #             best_results_and_thresholds_per_metric[metric][eval_name][1]
    #         )
    #         for metric in metrics
    #     }
    #     for eval_name in best_results_and_thresholds_per_metric[metrics[0]].keys()
    # }

    best_results_and_thresholds: dict[str, tuple[EvaluationResult, float | None]] = get_best_results_and_thresholds(
        dataset_evaluations, dataset_eval_names, "f1", append_mean_semsim_eval_results=True
    )

    # Create a list to hold the table data
    table_data = []

    # Process the evaluations and results
    # for eval_name, metrics_and_thresholds in best_scores_and_thresholds_per_eval_name.items():
    #     row = [prettify_eval_name(eval_name)]
    #     for metric in metrics:
    #         if metric not in metrics_and_thresholds:
    #             raise ValueError(f"Metric '{metric}' not found in results for evaluation '{eval_name}'")
    #
    #         score, threshold = metrics_and_thresholds[metric]
    #         row.append(f"{score:.2f}" + (f" ({threshold})" if threshold is not None else " (-)"))
    #
    #     # Append the data to the table list
    #     table_data.append(row)

    for eval_name, result_and_threshold in best_results_and_thresholds.items():
        row = [insert_latex_column_breaks(prettify_eval_name(eval_name))]

        result, threshold = result_and_threshold
        row.append(f"{threshold:.2f}" if threshold is not None else "-")
        for metric in metrics:
            row.append(f"{getattr(result, metric):.3f}")

        row.append("+/- SD" if "mean" in eval_name else "-")

        # Append the data to the table list
        table_data.append(row)

    # Create a DataFrame
    df = pd.DataFrame(table_data)
    # columns=["Evaluation Run Name", "ST", "Precision", "Recall", "F1 Score"]

    # Convert the DataFrame to a LaTeX table
    latex_table: str = df.to_latex(index=False, header=False)

    file_path: Path = Path(__file__).parent / "result_tables" / f"resul_table_{dataset_eval_names[0]}.tex"
    file_path.write_text(latex_table)

    logger.info(f"Results table written to: {file_path}")
    logger.info("-" * 80)

    logger.info(f"\n{latex_table}")


# make_results_table(
#     dataset_name="1-2_pred_wildcard_subsample-2000",
#     dataset_eval_names=[
#         "1-1_original-pattern",
#         "2-1_pred_semsim-fix_wildcard_cn",
#         "2-1_pred_semsim-fix_wildcard_w2v",
#         "2-2_pred_semsim-fix-lemma_wildcard_cn",
#         "2-2_pred_semsim-fix-lemma_wildcard_w2v",
#     ],
# )

semsim_ctx_eval_names = [
    "2-3_pred_semsim-ctx_wildcard_e5_nref-1",
    "2-3_pred_semsim-ctx_wildcard_e5_nref-10",
    "2-3_pred_semsim-ctx_wildcard_e5-at_nref-1",
    "2-3_pred_semsim-ctx_wildcard_e5-at_nref-10",
    "2-3_pred_semsim-ctx_wildcard_gte_nref-1",
    "2-3_pred_semsim-ctx_wildcard_gte_nref-10",
    "2-3_pred_semsim-ctx_wildcard_gte-at_nref-1",
    "2-3_pred_semsim-ctx_wildcard_gte-at_nref-10"
]
for semsim_ctxt_eval_name in semsim_ctx_eval_names:
    make_results_table(
        dataset_name="1-2_pred_wildcard_subsample-2000",
        dataset_eval_names=[semsim_ctxt_eval_name],
    )


