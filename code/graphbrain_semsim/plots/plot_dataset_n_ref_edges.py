from collections import defaultdict
from pathlib import Path
import re

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from graphbrain_semsim import logger, PLOT_DIR
from graphbrain_semsim.datasets.evaluate_dataset import DatasetEvaluation
from graphbrain_semsim.datasets.models import EvaluationResult
from graphbrain_semsim.eval_tools.result_data.dataset_evals import (
    get_dataset_evaluations, get_best_results_and_thresholds
)
from graphbrain_semsim.eval_tools.utils.pretty_names import prettify_eval_name
from graphbrain_semsim.plots import plot_base_config
from graphbrain_semsim.plots._base_config import (
    get_plot_line_color, PLOT_LINE_WEIGHTS, PLOT_LINE_STYLES, EVAL_METRIC_LABELS
)

from graphbrain_semsim.case_studies.conflicts.config import CASE_STUDY

plot_base_config()

PLOT_SUB_DIR_NAME: str = "dataset_n_ref_edges"


def plot(
        case_study: str,
        dataset_name: str,
        dataset_eval_names: list[str],
        selection_eval_metric: str,
        comparison_eval_metric: str,
):
    """
    Plot the eval metrics for the given dataset evaluations versus the number of reference edges.
    """
    dataset_id: str = f"dataset_{case_study}_{dataset_name}"

    logger.info(
        f"Making number of ref edges plot for dataset '{dataset_id}' and pattern configs:\n"
        + "\n".join([f" - {dataset_eval_name}" for dataset_eval_name in dataset_eval_names])
    )

    plot_data: dict[str, list[tuple[int, float]]] = collect_evaluation_run_category_data(
        case_study, dataset_id, dataset_eval_names, selection_eval_metric, comparison_eval_metric
    )

    # plot the data
    figure: Figure = Figure(figsize=(10, 8))
    axes: Axes = figure.add_axes(
        (0, 0, 1, 1), xlabel="Number of Reference Edges (N)", ylabel=EVAL_METRIC_LABELS[comparison_eval_metric]
    )
    # axes.set_title()

    for eval_run_cat_name, cat_plot_data in plot_data.items():
        # sort the data by the number of reference edges
        cat_plot_data.sort(key=lambda x: x[0])
        # n_refs, eval_metric_values = zip(*cat_plot_data)
        n_refs, eval_metric_values, eval_metric_std_devs = zip(*cat_plot_data)

        pattern_and_model: str = eval_run_cat_name.split(" - ")[0]
        plot_line_color: str = get_plot_line_color(pattern_and_model)
        # plot_line_weight: str = get_plot_line_weight(eval_run_cat_name)

        axes.plot(
            n_refs, eval_metric_values,
            label=prettify_eval_name(eval_run_cat_name),
            marker='o',
            color=plot_line_color,
            linestyle=PLOT_LINE_STYLES[comparison_eval_metric],
            # **PLOT_LINE_WEIGHTS[plot_line_weight]
            **PLOT_LINE_WEIGHTS["bold"]
        )

        eval_metric_lower_bound = [v - s for v, s in zip(eval_metric_values, eval_metric_std_devs)]
        eval_metric_upper_bound = [v + s for v, s in zip(eval_metric_values, eval_metric_std_devs)]

        std_dev_label: str = prettify_eval_name(eval_run_cat_name).replace("mean", "std-dev")

        axes.fill_between(
            n_refs, eval_metric_lower_bound, eval_metric_upper_bound,
            label=std_dev_label,
            color=plot_line_color,
            # **PLOT_LINE_WEIGHTS["light"]
            alpha=0.3
        )

    axes.legend(loc='upper left', bbox_to_anchor=(1, 1))

    plot_file_name: str = f"{dataset_id}_best-{selection_eval_metric}_nref_vs_{comparison_eval_metric}.png"
    plot_file_path: Path = PLOT_DIR / PLOT_SUB_DIR_NAME / plot_file_name
    figure.savefig(plot_file_path, bbox_inches='tight')
    logger.info(f"Plot saved to '{plot_file_path}'")


def collect_evaluation_run_category_data(
        case_study: str,
        dataset_id: str,
        dataset_eval_names: list[str],
        selection_eval_metric: str,
        comparison_eval_metric: str,
) -> dict[str, list[tuple[int, float]]]:
    # plot data is a dict mapping from eval run category to a list of tuples of the form:
    # eval run category -> (n_ref_edges, eval_metric_value)
    eval_run_cat_data: dict[str, list[tuple[int, float]]] = defaultdict(list)

    # the loop is constructed this way to avoid loading all dataset evaluations at once
    for dataset_eval_name in dataset_eval_names:
        dataset_evaluations: list[list[DatasetEvaluation]] = get_dataset_evaluations(
            [dataset_eval_name], case_study, dataset_id
        )

        # Check that all dataset evaluations actually have reference edges
        assert all(
            all(sub_evaluation.ref_edges is not None for sub_evaluation in sub_evaluations)
            for sub_evaluations in dataset_evaluations
        )

        # For each dataset evaluation, get the best evaluation result considering the selection_eval_metric
        best_results_and_thresholds: dict[str, tuple[EvaluationResult, float | None]] = get_best_results_and_thresholds(
            dataset_evaluations, [dataset_eval_name], selection_eval_metric, add_mean_eval_results=True
        )

        best_results_and_n_refs: dict[str, tuple[EvaluationResult, int]] = {
            eval_name: (eval_result, get_n_ref_edges_from_eval_name(eval_name))
            for eval_name, (eval_result, _) in best_results_and_thresholds.items()
        }

        for eval_name, (eval_result, n_refs) in best_results_and_n_refs.items():
            # # skip the mean eval result used for plotting metrics vs threshold
            # if "mean" in eval_name and "best" not in eval_name:
            #     continue
            #
            # eval_run_cat_data[get_eval_run_category(eval_name)].append(
            #     (n_refs, getattr(eval_result, comparison_eval_metric))
            # )

            if "mean-best" in eval_name:
                eval_run_cat_data[get_eval_run_category(eval_name)].append(
                    (
                        n_refs,
                        getattr(eval_result, comparison_eval_metric),
                        getattr(eval_result.std_dev, comparison_eval_metric)
                    )
                )

    return eval_run_cat_data


def get_plot_line_weight(eval_run_category: str) -> str:
    return "bold" if "mean-best" in eval_run_category else "light"


def get_eval_run_category(dataset_eval_name: str) -> str:
    # Get everything before the '-nref' substring
    pattern_and_model: str = dataset_eval_name.split("_nref")[0]
    # Get everything after the last underscore
    s_mod_or_mean_best: str = dataset_eval_name.split("_")[-1]
    return f"{pattern_and_model} - {s_mod_or_mean_best}"


def get_n_ref_edges_from_eval_name(dataset_eval_name: str) -> int:
    # Get the numer of reference edges from the dataset evaluation name
    # Find the 'nref-' substring and get the number that follows it via regex
    match = re.search(r"nref-(\d+)", dataset_eval_name)
    assert match is not None, f"Could not find 'nref-' substring in: '{dataset_eval_name}'"
    return int(match.group(1))


if __name__ == "__main__":
    plot(
        case_study=CASE_STUDY,
        dataset_name="1-2_pred_wildcard_subsample-2000",
        dataset_eval_names=[
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
        selection_eval_metric="f1",
        comparison_eval_metric="f1"
    )
