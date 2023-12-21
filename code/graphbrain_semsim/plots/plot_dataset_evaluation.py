from pathlib import Path
from statistics import mean
from typing import Optional

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pydantic import BaseModel

from graphbrain_semsim import logger, PLOT_DIR
from graphbrain_semsim.datasets.evaluate_dataset import DatasetEvaluation, EVALUATION_FILE_SUFFIX
from graphbrain_semsim.datasets.get_evaluations import get_dataset_evaluations
from graphbrain_semsim.datasets.models import EvaluationResult
from graphbrain_semsim.plots import plot_base_config
from graphbrain_semsim.utils.general import all_equal

from graphbrain_semsim.case_studies.conflicts.config import CASE_STUDY

PLOT_DIR_NAME: str = "dataset_evaluation"

PLOT_LINE_COLORS: list[str] = ['red', 'blue', 'orange', 'purple', 'cyan', 'green', 'magenta', 'yellow', 'brown', 'pink']
PLOT_LINE_STYLES: dict[str, str] = {'precision': 'dotted', 'recall': 'dashed', 'f1': 'dashdot', 'accuracy': 'solid'}
PLOT_LINE_WEIGHTS: dict[str, dict[str, str | float]] = {
    'bold': {
        'linewidth': '3',
        'alpha': 1.0
    },
    'light': {
        'linewidth': '1',
        'alpha': 0.5
    },
}

plot_base_config()


class DatasetEvaluationPlotInfo(BaseModel):
    dataset_eval_id: str
    dataset_eval_name: str
    dataset_evaluation: DatasetEvaluation
    plot_line_weight: Optional[str] = None
    plot_line_color: Optional[int] = None


def plot(
        case_study: str,
        dataset_name: str,
        dataset_eval_names: list[str],
        eval_metrics: list[str],
        plot_name_suffix: str,
):
    """
    Plot the evaluation results for the given dataset.
    Plot precision, recall and F1 curves for the lemma matcher and for the SemSim matcher.
    The lemma matcher is the baseline (plotted as a dashed line).
    The SemSim matcher is plotted as a solid line for each threshold value.
    """
    dataset_id: str = f"dataset_{case_study}_{dataset_name}"

    assert len(dataset_eval_names) <= len(PLOT_LINE_COLORS), (
        f"Number of dataset evaluations ({len(dataset_eval_names)}) must be less than or equal to "
        f"the number of specified colors ({len(PLOT_LINE_COLORS)})"
    )

    dataset_eval_plot_infos: list[DatasetEvaluationPlotInfo] = get_dataset_eval_plot_infos(
        dataset_eval_names, case_study, dataset_id
    )
    logger.info(
        f"Making dataset evaluation plot for dataset '{dataset_id}' and pattern configs:\n"
        + "\n".join([f" - {dataset_eval_name}" for dataset_eval_name in dataset_eval_names])
    )

    figure: Figure = Figure(figsize=(10, 7))
    axes: Axes = figure.add_axes((0, 0, 1, 1), xlabel="similarity threshold", ylabel="evaluation metric")

    for dataset_eval_plot_infos_info in dataset_eval_plot_infos:
        plot_dataset_evaluation(dataset_eval_plot_infos_info, axes, eval_metrics)

    axes.legend(loc='upper left', bbox_to_anchor=(1.04, 1), borderaxespad=0)
    axes.set_title(f"Evaluation of {dataset_name}")

    plot_file_name: str = f"{dataset_id}_{EVALUATION_FILE_SUFFIX}_{plot_name_suffix}_{'-'.join(eval_metrics)}.png"
    plot_file_path: Path = PLOT_DIR / PLOT_DIR_NAME / plot_file_name
    figure.savefig(plot_file_path, bbox_inches='tight')
    logger.info(f"Plot saved to '{plot_file_path}'")


def get_dataset_eval_plot_infos(
        dataset_eval_names: list[str],
        case_study: str,
        dataset_id: str
) -> list[DatasetEvaluationPlotInfo]:
    dataset_evaluations: dict[int, list[DatasetEvaluation]] = get_dataset_evaluations(
        dataset_eval_names, case_study, dataset_id
    )
    dataset_eval_plot_infos: list[DatasetEvaluationPlotInfo] = []

    for dataset_eval_idx, dataset_sub_evals in dataset_evaluations.items():
        dataset_eval_name: str = dataset_eval_names[dataset_eval_idx]
        dataset_eval_id: str = f"{dataset_id}_{EVALUATION_FILE_SUFFIX}_{case_study}_{dataset_eval_name}"

        if len(dataset_sub_evals) > 1:
            dataset_eval_plot_infos += process_sub_evaluations(
                dataset_sub_evals, dataset_eval_name, dataset_eval_id, dataset_eval_idx
            )
        else:
            dataset_eval_plot_infos.append(DatasetEvaluationPlotInfo(
                dataset_eval_id=dataset_eval_id,
                dataset_eval_name=dataset_eval_name,
                dataset_evaluation=dataset_sub_evals[0],
                plot_line_weight='bold',
                plot_line_color=dataset_eval_idx,
            ))
    return dataset_eval_plot_infos


def process_sub_evaluations(
        sub_evaluations: list[DatasetEvaluation],
        dataset_eval_name: str,
        dataset_eval_id: str,
        dataset_eval_idx: int
) -> list[DatasetEvaluationPlotInfo]:
    assert all(sub_evaluation.semsim_eval_results for sub_evaluation in sub_evaluations)
    assert all_equal(sub_evaluation.semsim_eval_results.keys() for sub_evaluation in sub_evaluations)

    # compute mean values for each threshold
    mean_semsim_eval_scores: dict[float, EvaluationResult] = {
        t: EvaluationResult(
            accuracy=mean([sub_eval.semsim_eval_results[t].accuracy for sub_eval in sub_evaluations]),
            precision=mean([sub_eval.semsim_eval_results[t].precision for sub_eval in sub_evaluations]),
            recall=mean([sub_eval.semsim_eval_results[t].recall for sub_eval in sub_evaluations]),
            f1=mean([sub_eval.semsim_eval_results[t].f1 for sub_eval in sub_evaluations]),
        )
        for t in sub_evaluations[0].semsim_eval_results.keys()
    }

    mean_dataset_evaluation: DatasetEvaluation = sub_evaluations[0].model_copy(update={
        "semsim_eval_results": mean_semsim_eval_scores
    })

    sub_eval_plot_infos: list[DatasetEvaluationPlotInfo] = [
        DatasetEvaluationPlotInfo(
            dataset_eval_id=f"{dataset_eval_id}_{sub_idx}",
            dataset_eval_name=f"{dataset_eval_name}_{sub_idx}",
            dataset_evaluation=sub_evaluation,
            plot_line_weight='light',
            plot_line_color=dataset_eval_idx,
        )
        for sub_idx, sub_evaluation in enumerate(sub_evaluations)
    ]
    sub_eval_plot_infos.append(DatasetEvaluationPlotInfo(
        dataset_eval_id=f"{dataset_eval_id}_mean",
        dataset_eval_name=f"{dataset_eval_name}_mean",
        dataset_evaluation=mean_dataset_evaluation,
        plot_line_weight='bold',
        plot_line_color=dataset_eval_idx,
    ))
    return sub_eval_plot_infos


def plot_dataset_evaluation(
    dataset_eval_plot_info: DatasetEvaluationPlotInfo,
    axes: Axes,
    eval_metrics: list[str]
):
    if dataset_eval_plot_info.dataset_evaluation.semsim_eval_results:
        thresholds: list[float] = list(dataset_eval_plot_info.dataset_evaluation.semsim_eval_results.keys())
        eval_scores: list[EvaluationResult] = list(dataset_eval_plot_info.dataset_evaluation.semsim_eval_results.values())
    else:
        thresholds: list[float] = [0.0, 1.0]  # mock thresholds for plotting
        eval_scores: list[EvaluationResult] = [dataset_eval_plot_info.dataset_evaluation.symbolic_eval_results] * len(thresholds)

    for eval_metric in eval_metrics:
        axes.plot(
            thresholds, [getattr(eval_score, eval_metric) for eval_score in eval_scores],
            label=f"{dataset_eval_plot_info.dataset_eval_id} - {eval_metric.capitalize()}",
            linestyle=PLOT_LINE_STYLES[eval_metric],
            color=PLOT_LINE_COLORS[dataset_eval_plot_info.plot_line_color],
            **PLOT_LINE_WEIGHTS[dataset_eval_plot_info.plot_line_weight],
        )


if __name__ == "__main__":
    plot(
        case_study=CASE_STUDY,
        dataset_name="1-1_wildcard_preds_subsample-2000",
        dataset_eval_names=[
            "1-1_original-pattern",
            "2-1_preds_semsim-fix_wildcard",
            "2-2_preds_semsim-fix-lemma_wildcard",
        ],
        eval_metrics=["precision", "recall"],
        plot_name_suffix="original_vs_semsim-fix"
    )
    plot(
        case_study=CASE_STUDY,
        dataset_name="1-1_wildcard_preds_subsample-2000",
        dataset_eval_names=[
            "1-1_original-pattern",
            "2-1_preds_semsim-fix_wildcard",
            "2-2_preds_semsim-fix-lemma_wildcard",
        ],
        eval_metrics=["f1"],
        plot_name_suffix="original_vs_semsim-fix"
    )
    plot(
        case_study=CASE_STUDY,
        dataset_name="1-1_wildcard_preds_subsample-2000",
        dataset_eval_names=[
            "1-1_original-pattern",
            "2-1_preds_semsim-fix_wildcard",
            "2-2_preds_semsim-fix-lemma_wildcard",
        ],
        eval_metrics=["accuracy"],
        plot_name_suffix="original_vs_semsim-fix"
    )
    plot(
        case_study=CASE_STUDY,
        dataset_name="1-1_wildcard_preds_subsample-2000",
        dataset_eval_names=[
            "1-1_original-pattern",
            "2-2_preds_semsim-fix-lemma_wildcard",
            "2-3_preds_semsim-ctx_wildcard_nref-10"
        ],
        eval_metrics=["precision", "recall"],
        plot_name_suffix="original_vs_semsim-fix_vs_semsim-ctx_nref-10"
    )
    plot(
        case_study=CASE_STUDY,
        dataset_name="1-1_wildcard_preds_subsample-2000",
        dataset_eval_names=[
            "1-1_original-pattern",
            "2-2_preds_semsim-fix-lemma_wildcard",
            "2-3_preds_semsim-ctx_wildcard_nref-10"
        ],
        eval_metrics=["f1"],
        plot_name_suffix="original_vs_semsim-fix_vs_semsim-ctx_nref-10"
    )
    plot(
        case_study=CASE_STUDY,
        dataset_name="1-1_wildcard_preds_subsample-2000",
        dataset_eval_names=[
            "1-1_original-pattern",
            "2-2_preds_semsim-fix-lemma_wildcard",
            "2-3_preds_semsim-ctx_wildcard_nref-10"
        ],
        eval_metrics=["accuracy"],
        plot_name_suffix="original_vs_semsim-fix_vs_semsim-ctx_nref-10"
    )
    plot(
        case_study=CASE_STUDY,
        dataset_name="1-1_wildcard_preds_subsample-2000",
        dataset_eval_names=[
            "1-1_original-pattern",
            "2-2_preds_semsim-fix-lemma_wildcard",
            "2-3_preds_semsim-ctx_wildcard_nref-1"
        ],
        eval_metrics=["precision", "recall"],
        plot_name_suffix="original_vs_semsim-fix_vs_semsim-ctx_nref-1"
    )
    plot(
        case_study=CASE_STUDY,
        dataset_name="1-1_wildcard_preds_subsample-2000",
        dataset_eval_names=[
            "1-1_original-pattern",
            "2-2_preds_semsim-fix-lemma_wildcard",
            "2-3_preds_semsim-ctx_wildcard_nref-1"
        ],
        eval_metrics=["f1"],
        plot_name_suffix="original_vs_semsim-fix_vs_semsim-ctx_nref-1"
    )
    plot(
        case_study=CASE_STUDY,
        dataset_name="1-1_wildcard_preds_subsample-2000",
        dataset_eval_names=[
            "2-3_preds_semsim-ctx_wildcard_nref-10"
        ],
        eval_metrics=["precision", "recall", "f1"],
        plot_name_suffix="semsim-ctx_nref-10"
    )
