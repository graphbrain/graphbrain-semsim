from pathlib import Path
from statistics import mean
from typing import Optional

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pydantic import BaseModel

from graphbrain_semsim import logger, PLOT_DIR
from graphbrain_semsim.datasets.config import DATASET_EVAL_DIR
from graphbrain_semsim.datasets.evaluate_dataset import DatasetEvaluation, EVALUATION_FILE_SUFFIX
from graphbrain_semsim.datasets.models import EvaluationScore
from graphbrain_semsim.eval_tools.plots import plot_base_config
from graphbrain_semsim.utils.general import load_json, all_equal

from graphbrain_semsim.case_studies.conflicts.config import CASE_STUDY

PLOT_DIR_NAME: str = "dataset_evaluation"

PLOT_LINE_COLORS: list[str] = ['red', 'blue', 'orange', 'purple', 'cyan', 'green', 'magenta', 'yellow', 'brown', 'pink']
PLOT_LINE_MARKER: dict[str, str] = {'precision': 'o', 'recall': 's', 'f1': '^'}
PLOT_LINE_STYLES: dict[str, dict[str, str | float]] = {
    'bold': {
        'linewidth': '3',
        'linestyle': '-',
        'alpha': 1.0
    },
    'light': {
        'linewidth': '1',
        'linestyle': '-',
        'alpha': 0.5
    },
    'dashed': {
        'linewidth': '1',
        'linestyle': '--',
        'alpha': 1.0
    }
}

plot_base_config()


class DatasetEvaluationPlotInfo(BaseModel):
    dataset_eval_id: str
    dataset_evaluation: DatasetEvaluation
    plot_line_style: Optional[str] = None
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

    dataset_eval_plot_infos: list[DatasetEvaluationPlotInfo] = []
    for dataset_eval_idx, dataset_eval_name in enumerate(dataset_eval_names):
        dataset_eval_id: str = f"{dataset_id}_{EVALUATION_FILE_SUFFIX}_{case_study}_{dataset_eval_name}"
        dataset_evaluation_sub_dir: Path = DATASET_EVAL_DIR / dataset_eval_id
        if dataset_evaluation_sub_dir.is_dir():
            dataset_eval_plot_infos += process_sub_evaluations(
                dataset_evaluation_sub_dir, dataset_eval_id, dataset_eval_idx
            )
        else:
            dataset_evaluation_file_path: Path = DATASET_EVAL_DIR / f"{dataset_eval_id}.json"
            dataset_evaluation: DatasetEvaluation = load_json(
                dataset_evaluation_file_path, DatasetEvaluation, exit_on_error=True
            )
            dataset_eval_plot_infos.append(DatasetEvaluationPlotInfo(
                dataset_eval_id=dataset_eval_id,
                dataset_evaluation=dataset_evaluation,
                plot_line_style='bold',
                plot_line_color=dataset_eval_idx,
            ))

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


def process_sub_evaluations(
        dataset_evaluation_sub_dir: Path, dataset_eval_id: str, dataset_eval_idx: int
) -> list[DatasetEvaluationPlotInfo]:
    sub_evaluations: list[DatasetEvaluation] = [
        load_json(sub_evaluation_file_path, DatasetEvaluation, exit_on_error=True)
        for sub_evaluation_file_path in dataset_evaluation_sub_dir.iterdir()
    ]

    assert all(sub_evaluation.semsim_eval_scores for sub_evaluation in sub_evaluations)
    assert all_equal(sub_evaluation.semsim_eval_scores.keys() for sub_evaluation in sub_evaluations)

    # compute mean values for each threshold
    mean_semsim_eval_scores: dict[float, EvaluationScore] = {
        t: EvaluationScore(
            precision=mean([sub_eval.semsim_eval_scores[t].precision for sub_eval in sub_evaluations]),
            recall=mean([sub_eval.semsim_eval_scores[t].recall for sub_eval in sub_evaluations]),
            f1=mean([sub_eval.semsim_eval_scores[t].f1 for sub_eval in sub_evaluations]),
        )
        for t in sub_evaluations[0].semsim_eval_scores.keys()
    }

    mean_dataset_evaluation: DatasetEvaluation = sub_evaluations[0].model_copy(update={
        "semsim_eval_scores": mean_semsim_eval_scores
    })

    sub_eval_plot_infos: list[DatasetEvaluationPlotInfo] = [
        DatasetEvaluationPlotInfo(
            dataset_eval_id=f"{dataset_eval_id}_{sub_idx}",
            dataset_evaluation=sub_evaluation,
            plot_line_style='light',
            plot_line_color=dataset_eval_idx,
        )
        for sub_idx, sub_evaluation in enumerate(sub_evaluations)
    ]
    sub_eval_plot_infos.append(DatasetEvaluationPlotInfo(
        dataset_eval_id=f"{dataset_eval_id}_mean",
        dataset_evaluation=mean_dataset_evaluation,
        plot_line_style='bold',
        plot_line_color=dataset_eval_idx,
    ))
    return sub_eval_plot_infos


def plot_dataset_evaluation(
    dataset_eval_plot_info: DatasetEvaluationPlotInfo,
    axes: Axes,
    eval_metrics: list[str]
):
    if dataset_eval_plot_info.dataset_evaluation.semsim_eval_scores:
        thresholds: list[float] = list(dataset_eval_plot_info.dataset_evaluation.semsim_eval_scores.keys())
        eval_scores: list[EvaluationScore] = list(dataset_eval_plot_info.dataset_evaluation.semsim_eval_scores.values())
    else:
        thresholds: list[float] = [0.0, 0.5, 1.0]
        eval_scores: list[EvaluationScore] = [dataset_eval_plot_info.dataset_evaluation.symbolic_eval_score] * len(thresholds)

    if 'precision' in eval_metrics:
        axes.plot(
            thresholds, [eval_score.precision for eval_score in eval_scores],
            label=f"{dataset_eval_plot_info.dataset_eval_id} - Precision",
            marker=PLOT_LINE_MARKER['precision'],
            color=PLOT_LINE_COLORS[dataset_eval_plot_info.plot_line_color],
            **PLOT_LINE_STYLES[dataset_eval_plot_info.plot_line_style],
        )
    if 'recall' in eval_metrics:
        axes.plot(
            thresholds, [eval_score.recall for eval_score in eval_scores],
            label=f"{dataset_eval_plot_info.dataset_eval_id} - Recall",
            marker=PLOT_LINE_MARKER['recall'],
            color=PLOT_LINE_COLORS[dataset_eval_plot_info.plot_line_color],
            **PLOT_LINE_STYLES[dataset_eval_plot_info.plot_line_style],
        )
    if 'f1' in eval_metrics:
        axes.plot(
            thresholds, [eval_score.f1 for eval_score in eval_scores],
            label=f"{dataset_eval_plot_info.dataset_eval_id} - F1",
            marker=PLOT_LINE_MARKER['f1'],
            color=PLOT_LINE_COLORS[dataset_eval_plot_info.plot_line_color],
            **PLOT_LINE_STYLES[dataset_eval_plot_info.plot_line_style],
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
            "2-2_preds_semsim-fix-lemma_wildcard",
            "2-3_preds_semsim-ctx_wildcard_nref-10"
        ],
        eval_metrics=["precision", "recall"],
        plot_name_suffix="semsim-fix_vs_semsim-ctx"
    )
    plot(
        case_study=CASE_STUDY,
        dataset_name="1-1_wildcard_preds_subsample-2000",
        dataset_eval_names=[
            "2-2_preds_semsim-fix-lemma_wildcard",
            "2-3_preds_semsim-ctx_wildcard_nref-10"
        ],
        eval_metrics=["f1"],
        plot_name_suffix="semsim-fix_vs_semsim-ctx"
    )
    plot(
        case_study=CASE_STUDY,
        dataset_name="1-1_wildcard_preds_subsample-2000",
        dataset_eval_names=[
            "1-1_original-pattern",
            "2-3_preds_semsim-ctx_wildcard_nref-1"
        ],
        eval_metrics=["precision", "recall"],
        plot_name_suffix="original_vs_semsim-ctx"
    )
    plot(
        case_study=CASE_STUDY,
        dataset_name="1-1_wildcard_preds_subsample-2000",
        dataset_eval_names=[
            "1-1_original-pattern",
            "2-3_preds_semsim-ctx_wildcard_nref-1"
        ],
        eval_metrics=["f1"],
        plot_name_suffix="original_vs_semsim-ctx"
    )
