from pathlib import Path

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from graphbrain_semsim import logger, PLOT_DIR
from graphbrain_semsim.eval_tools.datasets.config import DATA_DIR
from graphbrain_semsim.eval_tools.datasets.evaluate_dataset import DatasetEvaluation, EVALUATION_FILE_SUFFIX
from graphbrain_semsim.eval_tools.plots import plot_base_config
from graphbrain_semsim.utils.general import load_json

PLOT_DIR_NAME: str = "dataset_evaluation"

plot_base_config()


def plot(dataset_name: str):
    """
    Plot the evaluation results for the given dataset.
    Plot precision, recall and F1 curves for the lemma matcher and for the SemSim matcher.
    The lemma matcher is the baseline (plotted as a dashed line).
    The SemSim matcher is plotted as a solid line for each threshold value.
    """
    dataset_evaluation_name: str = f"{dataset_name}_{EVALUATION_FILE_SUFFIX}"
    dataset_evaluation: DatasetEvaluation = load_json(DATA_DIR / f"{dataset_evaluation_name}.json", DatasetEvaluation)

    logger.info(f"Making plot '{dataset_evaluation_name}'...")

    figure: Figure = Figure(figsize=(10, 7))
    axes: Axes = figure.add_axes((0, 0, 1, 1), xlabel="similarity threshold", ylabel="evaluation metric")

    # Plotting lemma matcher (baseline) with dashed lines
    if dataset_evaluation.lemma_match_eval_score:
        axes.axhline(
            y=dataset_evaluation.lemma_match_eval_score.precision, color='b', linestyle='--',
            label='Lemma Matcher Precision'
        )
        axes.axhline(
            y=dataset_evaluation.lemma_match_eval_score.recall, color='orange', linestyle='--',
            label='Lemma Matcher Recall'
        )
        axes.axhline(
            y=dataset_evaluation.lemma_match_eval_score.f1, color='purple', linestyle='--',
            label='Lemma Matcher F1'
        )

    # Plotting SemSim matcher with solid lines
    if dataset_evaluation.semsim_match_eval_scores:
        thresholds = sorted(dataset_evaluation.semsim_match_eval_scores.keys())

        precisions = [dataset_evaluation.semsim_match_eval_scores[t].precision for t in thresholds]
        recalls = [dataset_evaluation.semsim_match_eval_scores[t].recall for t in thresholds]
        f1_scores = [dataset_evaluation.semsim_match_eval_scores[t].f1 for t in thresholds]

        axes.plot(thresholds, precisions, color='b', linestyle='-', label='SemSim Matcher Precision')
        axes.plot(thresholds, recalls, color='orange', linestyle='-', label='SemSim Matcher Recall')
        axes.plot(thresholds, f1_scores, color='purple', linestyle='-', label='SemSim Matcher F1')

    # Set legend, title, and display the plot
    axes.legend(loc='best')
    axes.set_title(f"Evaluation of {dataset_name}")

    fig_file_path: Path = PLOT_DIR / PLOT_DIR_NAME / f"{dataset_evaluation_name}.png"
    figure.savefig(fig_file_path, bbox_inches='tight')
    logger.info(f"Plot saved to '{fig_file_path}'")


if __name__ == "__main__":
    plot("dataset_conflicts_1-1_wildcard_preds_subsample-2000_recreated")
