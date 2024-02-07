from pathlib import Path
import logging

import pandas as pd

from graphbrain_semsim.datasets.models import DatasetEvaluation, EvaluationResult
from graphbrain_semsim.eval_tools.result_data.dataset_evals import (
    get_dataset_evaluations, get_best_results_and_thresholds
)
from graphbrain_semsim import DATA_DIR
from graphbrain_semsim.case_studies.conflicts.config import CASE_STUDY
from graphbrain_semsim.eval_tools.utils.pretty_names import prettify_eval_name
# from graphbrain_semsim.utils.general import all_equal

logger = logging.getLogger(__name__)


def insert_latex_column_breaks(eval_name: str) -> str:
    eval_name = eval_name.replace(" ", " & ")

    required_num_breaks = 2
    num_breaks = eval_name.count("&")
    eval_name += " & " * (required_num_breaks - num_breaks)

    return eval_name


def make_results_table(
    dataset_name: str,
    dataset_eval_names: list[str],
    result_table_name_suffix: str,
    case_study: str = CASE_STUDY,
    metrics: tuple[str] = ('precision', 'recall', 'f1')
):
    dataset_id: str = f"dataset_{case_study}_{dataset_name}"

    # Create a list to hold the table data
    table_data = []

    # this loop is constructed this way to avoid loading all the dataset evaluations into memory at once
    for dataset_eval_name in dataset_eval_names:
        dataset_evaluations: list[list[DatasetEvaluation]] = get_dataset_evaluations(
            [dataset_eval_name], case_study, dataset_id
        )
        logger.info(f"Loaded dataset evaluations. Making results table...")

        best_results_and_thresholds: dict[str, tuple[EvaluationResult, float | None]] = get_best_results_and_thresholds(
            dataset_evaluations, [dataset_eval_name], "f1", add_mean_eval_results=True
        )

        best_eval_names_sorted: list[str] = list(sorted(best_results_and_thresholds.keys()))

        # for eval_name, result_and_threshold in best_results_and_thresholds.items():
        for eval_name in best_eval_names_sorted:
            row = [insert_latex_column_breaks(prettify_eval_name(eval_name))]

            result, threshold = best_results_and_thresholds[eval_name]

            row.append(f"{threshold:.2f}" if threshold is not None else "-")

            for metric in metrics:
                row.append(f"{getattr(result, metric):.3f}")

            if "mean" in eval_name:
                assert result.std_dev is not None
                std_dev_str = f"+/- {result.std_dev.f1:.3f}"
                # if "best" not in eval_name:
                #     std_dev_str = "\\textsuperscript{*}" + std_dev_str
                row.append(std_dev_str)
            else:
                row.append("-")

            # if "semsim-ctx" in eval_name and "mean-best" not in eval_name:
            #     row_textit = [f"\\textit{{{name_part}}}" for name_part in row[0].split(" & ")]
            #     row_textit += [f"\\textit{{{entry}}}" for entry in row[1:]]
            #     row = row_textit

            # Append the data to the table list
            table_data.append(row)

        # remove the dataset evaluations from memory
        # (this should be done by gc automatically)
        del dataset_evaluations

        # Create a DataFrame
        df = pd.DataFrame(table_data)
        # columns=["Evaluation Run Name", "ST", "Precision", "Recall", "F1 Score"]

        # Convert the DataFrame to a LaTeX table
        latex_table: str = df.to_latex(index=False, header=False)

        # file_path: Path = DATA_DIR / "result_tables" / f"resul_table_{result_table_name_suffix}.tex"
        file_path: Path = DATA_DIR / "result_tables" / f"result_table_{dataset_eval_name}.tex"
        file_path.write_text(latex_table)
        logger.info(f"Results table written to: {file_path}")
        logger.info("-" * 80)

        logger.info(f"\n{latex_table}")


make_results_table(
    dataset_name="1-2_pred_wildcard_subsample-2000",
    dataset_eval_names=[
        # "1-1_original-pattern",
        # "2-1_pred_semsim-fix_wildcard_cn",
        # "2-1_pred_semsim-fix_wildcard_w2v",
        # "2-2_pred_semsim-fix-lemma_wildcard_cn",
        # "2-2_pred_semsim-fix-lemma_wildcard_w2v",
        "2-3_pred_semsim-ctx_wildcard_e5_nref-1",
        "2-3_pred_semsim-ctx_wildcard_e5_nref-3",
        "2-3_pred_semsim-ctx_wildcard_e5_nref-10",
        "2-3_pred_semsim-ctx_wildcard_e5-at_nref-1",
        "2-3_pred_semsim-ctx_wildcard_e5-at_nref-3",
        "2-3_pred_semsim-ctx_wildcard_e5-at_nref-10",
        "2-3_pred_semsim-ctx_wildcard_gte_nref-1",
        "2-3_pred_semsim-ctx_wildcard_gte_nref-3",
        "2-3_pred_semsim-ctx_wildcard_gte_nref-10",
        "2-3_pred_semsim-ctx_wildcard_gte-at_nref-1",
        "2-3_pred_semsim-ctx_wildcard_gte-at_nref-3",
        "2-3_pred_semsim-ctx_wildcard_gte-at_nref-10"
    ],
    result_table_name_suffix="semsim-ctx",
)


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
