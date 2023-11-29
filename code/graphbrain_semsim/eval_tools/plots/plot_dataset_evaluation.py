from pathlib import Path

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from graphbrain_semsim import logger, PLOT_DIR
from graphbrain_semsim.datasets.config import DATASET_EVAL_DIR
from graphbrain_semsim.datasets.evaluate_dataset import DatasetEvaluation, EVALUATION_FILE_SUFFIX
from graphbrain_semsim.eval_tools.plots import plot_base_config
from graphbrain_semsim.utils.general import load_json

from graphbrain_semsim.case_studies.conflicts.config import CASE_STUDY

PLOT_DIR_NAME: str = "dataset_evaluation"

plot_base_config()


def plot(
        case_study: str,
        dataset_name: str,
        pattern_eval_config_names: list[str],
        eval_metrics: list[str],
        plot_name_suffix: str,
):
    """
    Plot the evaluation results for the given dataset.
    Plot precision, recall and F1 curves for the lemma matcher and for the SemSim matcher.
    The lemma matcher is the baseline (plotted as a dashed line).
    The SemSim matcher is plotted as a solid line for each threshold value.
    """
    dataset_evaluations: list[DatasetEvaluation] = []
    for pattern_eval_config_name in pattern_eval_config_names:
        dataset_evaluation_name: str = (
            f"{dataset_name}_{EVALUATION_FILE_SUFFIX}_{case_study}_{pattern_eval_config_name}"
        )
        dataset_evaluation_file_path: Path = DATASET_EVAL_DIR / f"{dataset_evaluation_name}.json"
        dataset_evaluations.append(load_json(dataset_evaluation_file_path, DatasetEvaluation, exit_on_error=True))

    logger.info(
        f"Making dataset evaluation plot for dataset '{dataset_name}' and pattern configs:\n"
        + "\n".join([f" - {pattern_eval_config_name}" for pattern_eval_config_name in pattern_eval_config_names])
    )

    figure: Figure = Figure(figsize=(10, 7))
    axes: Axes = figure.add_axes((0, 0, 1, 1), xlabel="similarity threshold", ylabel="evaluation metric")

    colors = ['red', 'blue', 'orange', 'purple', 'cyan', 'green', 'magenta', 'yellow', 'brown', 'pink']
    marker_types = {'precision': 'o', 'recall': '^', 'f1': 's'}  # Define marker types for each metric

    for index, dataset_evaluation in enumerate(dataset_evaluations):
        current_color = colors[index % len(colors)]  # Cycle through colors if more datasets than colors

        if dataset_evaluation.symbolic_eval_score:
            x_values = [0.0, 0.5, 1.0]

            if 'precision' in eval_metrics:
                axes.plot(
                    x_values, [dataset_evaluation.symbolic_eval_score.precision] * len(x_values), color=current_color,
                    linestyle='--',
                    marker=marker_types['precision'], label=f"{dataset_evaluation.pattern_eval_config_id} - Precision"
                )
            if 'recall' in eval_metrics:
                axes.plot(
                    x_values, [dataset_evaluation.symbolic_eval_score.recall] * len(x_values), color=current_color,
                    linestyle='--',
                    marker=marker_types['recall'], label=f"{dataset_evaluation.pattern_eval_config_id} - Recall"
                )
            if 'f1' in eval_metrics:
                axes.plot(
                    x_values, [dataset_evaluation.symbolic_eval_score.f1] * len(x_values), color=current_color,
                    linestyle='--',
                    marker=marker_types['f1'], label=f"{dataset_evaluation.pattern_eval_config_id} - F1"
                )

        if dataset_evaluation.semsim_eval_scores:
            thresholds = sorted(dataset_evaluation.semsim_eval_scores.keys())

            precisions = [dataset_evaluation.semsim_eval_scores[t].precision for t in thresholds]
            recalls = [dataset_evaluation.semsim_eval_scores[t].recall for t in thresholds]
            f1_scores = [dataset_evaluation.semsim_eval_scores[t].f1 for t in thresholds]

            if 'precision' in eval_metrics:
                axes.plot(
                    thresholds, precisions, color=current_color, linestyle='-',
                    marker=marker_types['precision'], label=f"{dataset_evaluation.pattern_eval_config_id} - Precision"
                )
            if 'recall' in eval_metrics:
                axes.plot(
                    thresholds, recalls, color=current_color, linestyle='-',
                    marker=marker_types['recall'], label=f"{dataset_evaluation.pattern_eval_config_id} - Recall"
                )
            if 'f1' in eval_metrics:
                axes.plot(
                    thresholds, f1_scores, color=current_color, linestyle='-',
                    marker=marker_types['f1'], label=f"{dataset_evaluation.pattern_eval_config_id} - F1"
                )

    axes.legend(loc='upper left', bbox_to_anchor=(1.04, 1), borderaxespad=0)
    axes.set_title(f"Evaluation of {dataset_name}")

    plot_file_name: str = f"{dataset_name}_{EVALUATION_FILE_SUFFIX}_{plot_name_suffix}_{'-'.join(eval_metrics)}.png"
    plot_file_path: Path = PLOT_DIR / PLOT_DIR_NAME / plot_file_name
    figure.savefig(plot_file_path, bbox_inches='tight')
    logger.info(f"Plot saved to '{plot_file_path}'")


if __name__ == "__main__":
    plot(
        case_study=CASE_STUDY,
        dataset_name="dataset_conflicts_1-1_wildcard_preds_subsample-2000",
        pattern_eval_config_names=[
            "1-1_original-pattern",
            "2-1_preds_semsim-fix_wildcard",
            "2-2_preds_semsim-fix-lemma_wildcard",
        ],
        eval_metrics=["precision", "recall"],
        plot_name_suffix="original_vs_semsim-fix"
    )
    plot(
        case_study=CASE_STUDY,
        dataset_name="dataset_conflicts_1-1_wildcard_preds_subsample-2000",
        pattern_eval_config_names=[
            "1-1_original-pattern",
            "2-1_preds_semsim-fix_wildcard",
            "2-2_preds_semsim-fix-lemma_wildcard",
        ],
        eval_metrics=["f1"],
        plot_name_suffix="original_vs_semsim-fix"
    )
    plot(
        case_study=CASE_STUDY,
        dataset_name="dataset_conflicts_1-1_wildcard_preds_subsample-2000",
        pattern_eval_config_names=[
            "2-2_preds_semsim-fix-lemma_wildcard",
            # "2-3_preds_semsim-ctx_wildcard",
            "2-3_preds_semsim-ctx_wildcard_nref-10_smod-0"
        ],
        eval_metrics=["precision", "recall"],
        plot_name_suffix="semsim-fix_vs_semsim-ctx"
    )
    plot(
        case_study=CASE_STUDY,
        dataset_name="dataset_conflicts_1-1_wildcard_preds_subsample-2000",
        pattern_eval_config_names=[
            "2-2_preds_semsim-fix-lemma_wildcard",
            # "2-3_preds_semsim-ctx_wildcard",
            "2-3_preds_semsim-ctx_wildcard_nref-10_smod-0"
        ],
        eval_metrics=["f1"],
        plot_name_suffix="semsim-fix_vs_semsim-ctx"
    )
    plot(
        case_study=CASE_STUDY,
        dataset_name="dataset_conflicts_1-1_wildcard_preds_subsample-2000",
        pattern_eval_config_names=[
            "1-1_original-pattern",
            # "2-3_preds_semsim-ctx_wildcard",
            "2-3_preds_semsim-ctx_wildcard_nref-10_smod-0"
        ],
        eval_metrics=["precision", "recall"],
        plot_name_suffix="original_vs_semsim-ctx"
    )
    plot(
        case_study=CASE_STUDY,
        dataset_name="dataset_conflicts_1-1_wildcard_preds_subsample-2000",
        pattern_eval_config_names=[
            "1-1_original-pattern",
            # "2-3_preds_semsim-ctx_wildcard",
            "2-3_preds_semsim-ctx_wildcard_nref-10_smod-0"
        ],
        eval_metrics=["f1"],
        plot_name_suffix="original_vs_semsim-ctx"
    )
