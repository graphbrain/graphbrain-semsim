from functools import lru_cache
from pathlib import Path

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from graphbrain_semsim import logger, PLOT_DIR, DATA_DIR
from graphbrain_semsim.datasets.evaluate_dataset import (
    DatasetEvaluation, EVALUATION_FILE_SUFFIX, get_positives_and_negatives
)
from graphbrain_semsim.datasets.models import EvaluationResult, LemmaDataset
from graphbrain_semsim.eval_tools.result_data.dataset_evals import (
    get_dataset_evaluations,
    compute_mean_semsim_eval_results
)
from graphbrain_semsim.eval_tools.utils.pretty_names import prettify_eval_name
from graphbrain_semsim.plots import plot_base_config
from graphbrain_semsim.plots._base_config import (
    PLOT_LINE_STYLES, PLOT_LINE_WEIGHTS, DatasetEvaluationPlotInfo, get_plot_line_color
)
from graphbrain_semsim.utils.file_handling import load_json
from graphbrain_semsim.utils.general import all_equal

from graphbrain_semsim.case_studies.conflicts.config import CASE_STUDY

plot_base_config()


PLOT_DIR_NAME: str = "dataset_evaluation"


def plot(
        case_study: str,
        dataset_name: str,
        dataset_eval_names: list[str],
        eval_metrics: list[str],
        plot_name_suffix: str,
        add_random_baseline: bool = False
):
    """
    Plot the evaluation results for the given dataset.
    Plot precision, recall and F1 curves for the lemma matcher and for the SemSim matcher.
    The lemma matcher is the baseline (plotted as a dashed line).
    The SemSim matcher is plotted as a solid line for each threshold value.
    """
    dataset_id: str = f"dataset_{case_study}_{dataset_name}"

    dataset_eval_plot_infos: list[DatasetEvaluationPlotInfo] = get_dataset_eval_plot_infos(
        dataset_eval_names, case_study, dataset_id
    )
    logger.info(
        f"Making dataset evaluation plot for dataset '{dataset_id}' and pattern configs:\n"
        + "\n".join([f" - {dataset_eval_name}" for dataset_eval_name in dataset_eval_names])
    )

    if add_random_baseline:
        dataset_eval_plot_infos.append(get_random_baseline_plot_info(dataset_id))

    figure: Figure = Figure(figsize=(10, 7))
    axes: Axes = figure.add_axes((0, 0, 1, 1), xlabel="similarity threshold", ylabel="evaluation metric")

    for dataset_eval_plot_infos_info in dataset_eval_plot_infos:
        plot_dataset_evaluation(dataset_eval_plot_infos_info, axes, eval_metrics)

    n_col = 1
    # n_col = 3 if not len(dataset_eval_plot_infos) == 2 else 2
    axes.legend(loc='upper left', bbox_to_anchor=(1, 1), ncol=n_col)
    # axes.set_title(f"Evaluation of {dataset_name}")

    plot_file_name: str = f"{dataset_id}_{EVALUATION_FILE_SUFFIX}_{plot_name_suffix}_{'-'.join(eval_metrics)}.png"
    plot_file_path: Path = PLOT_DIR / PLOT_DIR_NAME / plot_file_name
    figure.savefig(plot_file_path, bbox_inches='tight')
    logger.info(f"Plot saved to '{plot_file_path}'")


def get_dataset_eval_plot_infos(
        dataset_eval_names: list[str],
        case_study: str,
        dataset_id: str
) -> list[DatasetEvaluationPlotInfo]:
    dataset_evaluations: list[list[DatasetEvaluation]] = get_dataset_evaluations(
        dataset_eval_names, case_study, dataset_id
    )

    dataset_eval_plot_infos: list[DatasetEvaluationPlotInfo] = []
    for dataset_eval_idx, dataset_sub_evals in enumerate(dataset_evaluations):
        dataset_eval_name: str = dataset_eval_names[dataset_eval_idx]
        dataset_eval_id: str = f"{dataset_id}_{EVALUATION_FILE_SUFFIX}_{case_study}_{dataset_eval_name}"

        if len(dataset_sub_evals) > 1:
            dataset_eval_plot_infos += process_sub_evaluations(
                dataset_sub_evals, dataset_eval_name, dataset_eval_id
            )
        else:
            dataset_eval_plot_infos.append(DatasetEvaluationPlotInfo(
                dataset_eval_id=dataset_eval_id,
                dataset_eval_name=dataset_eval_name,
                dataset_evaluation=dataset_sub_evals[0],
                plot_line_color=get_plot_line_color(dataset_eval_id),
                plot_line_weight='bold',
            ))
    return dataset_eval_plot_infos


def process_sub_evaluations(
        sub_evaluations: list[DatasetEvaluation],
        dataset_eval_name: str,
        dataset_eval_id: str,
) -> list[DatasetEvaluationPlotInfo]:
    assert all(sub_evaluation.semsim_eval_results for sub_evaluation in sub_evaluations)
    assert all_equal(sub_evaluation.semsim_eval_results.keys() for sub_evaluation in sub_evaluations)

    mean_semsim_eval_scores = compute_mean_semsim_eval_results(sub_evaluations)
    mean_dataset_evaluation: DatasetEvaluation = sub_evaluations[0].model_copy(update={
        "semsim_eval_results": mean_semsim_eval_scores
    })

    sub_eval_plot_infos: list[DatasetEvaluationPlotInfo] = [
        DatasetEvaluationPlotInfo(
            dataset_eval_id=f"{dataset_eval_id}_{sub_idx + 1}",
            dataset_eval_name=f"{dataset_eval_name}_{sub_idx + 1}",
            dataset_evaluation=sub_evaluation,
            plot_line_color=get_plot_line_color(dataset_eval_id),
            plot_line_weight='light',

        )
        for sub_idx, sub_evaluation in enumerate(sub_evaluations)
    ]
    sub_eval_plot_infos.append(DatasetEvaluationPlotInfo(
        dataset_eval_id=f"{dataset_eval_id}_mean",
        dataset_eval_name=f"{dataset_eval_name}_mean",
        dataset_evaluation=mean_dataset_evaluation,
        plot_line_color=get_plot_line_color(dataset_eval_id),
        plot_line_weight='bold',

    ))
    return sub_eval_plot_infos


@lru_cache(maxsize=1)
def get_random_baseline_plot_info(dataset_id: str) -> DatasetEvaluationPlotInfo:
    dataset: LemmaDataset = load_json(DATA_DIR / "datasets" / f"{dataset_id}_recreated.json", LemmaDataset)
    dataset_positives, dataset_negatives = get_positives_and_negatives(dataset.all_lemma_matches)

    # Expected outcomes for a random classifier
    t_p = 0.5 * len(dataset_positives)
    f_p = 0.5 * len(dataset_negatives)
    t_n = 0.5 * len(dataset_negatives)
    f_n = 0.5 * len(dataset_positives)

    accuracy = (t_p + t_n) / dataset.n_samples
    precision = t_p / (t_p + f_p)
    recall = t_p / (t_p + f_n)
    f1_score = 2 * (precision * recall) / (precision + recall)

    random_dataset_evaluation_result = EvaluationResult(
        accuracy=accuracy, recall=recall, precision=precision, f1=f1_score
    )
    # logger.info(f"Random baseline evaluation result: {random_dataset_evaluation_result.model_dump_json(indent=2)}")

    return DatasetEvaluationPlotInfo(
        dataset_eval_id=f"{dataset_id}_random_baseline",
        dataset_eval_name="random_baseline",
        dataset_evaluation=DatasetEvaluation(
            dataset_id=dataset_id,
            pattern_eval_config_id="random_baseline",
            num_samples=dataset.n_samples,
            num_positive=t_p + f_p,
            num_negative=t_n + f_n,
            symbolic_eval_result=random_dataset_evaluation_result
        ),
        plot_line_color=get_plot_line_color(dataset_id),
        plot_line_weight='bold',
    )


def plot_dataset_evaluation(
    dataset_eval_plot_info: DatasetEvaluationPlotInfo,
    axes: Axes,
    eval_metrics: list[str]
):
    if dataset_eval_plot_info.dataset_evaluation.semsim_eval_results:
        thresholds: list[float] = list(dataset_eval_plot_info.dataset_evaluation.semsim_eval_results.keys())
        eval_scores: list[EvaluationResult] = (
            list(dataset_eval_plot_info.dataset_evaluation.semsim_eval_results.values())
        )
    else:
        thresholds: list[float] = [0.0, 1.0]  # mock thresholds for plotting
        eval_scores: list[EvaluationResult] = (
            [dataset_eval_plot_info.dataset_evaluation.symbolic_eval_result] * len(thresholds)
        )

    for eval_metric in eval_metrics:
        axes.plot(
            thresholds, [getattr(eval_score, eval_metric) for eval_score in eval_scores],
            label=f"{prettify_eval_name(dataset_eval_plot_info.dataset_eval_name)} - {eval_metric.capitalize()}",
            linestyle=PLOT_LINE_STYLES[eval_metric],
            color=dataset_eval_plot_info.plot_line_color,
            **PLOT_LINE_WEIGHTS[dataset_eval_plot_info.plot_line_weight],
        )


if __name__ == "__main__":
    # ----- SemSim FIX -----
    plot(
        case_study=CASE_STUDY,
        dataset_name="1-2_pred_wildcard_subsample-2000",
        dataset_eval_names=[
            "1-1_original-pattern",
            "2-1_pred_semsim-fix_wildcard_cn",
            "2-2_pred_semsim-fix-lemma_wildcard_cn",
        ],
        eval_metrics=["precision", "recall"],
        plot_name_suffix="original_vs_semsim-fix_cn_vs_semsim-fix-lemma_cn"
    )
    plot(
        case_study=CASE_STUDY,
        dataset_name="1-2_pred_wildcard_subsample-2000",
        dataset_eval_names=[
            "1-1_original-pattern",
            "2-1_pred_semsim-fix_wildcard_cn",
            "2-2_pred_semsim-fix-lemma_wildcard_cn",
        ],
        eval_metrics=["precision", "recall", "f1"],
        plot_name_suffix="original_vs_semsim-fix_cn_vs_semsim-fix-lemma_cn"
    )
    plot(
        case_study=CASE_STUDY,
        dataset_name="1-2_pred_wildcard_subsample-2000",
        dataset_eval_names=[
            "2-2_pred_semsim-fix-lemma_wildcard_w2v",
            "2-2_pred_semsim-fix-lemma_wildcard_cn",
        ],
        eval_metrics=["precision", "recall", "f1"],
        plot_name_suffix="semsim-fix-lemma_w2v_cn"
    )
    plot(
        case_study=CASE_STUDY,
        dataset_name="1-2_pred_wildcard_subsample-2000",
        dataset_eval_names=[
            "2-1_pred_semsim-fix_wildcard_cn",
            "2-2_pred_semsim-fix-lemma_wildcard_cn",
        ],
        eval_metrics=["precision", "recall", "f1"],
        plot_name_suffix="semsim-fix_cn_vs_semsim-fix-lemma_cn"
    )
    # ----- SemSim CTX - E5 -----
    plot(
        case_study=CASE_STUDY,
        dataset_name="1-2_pred_wildcard_subsample-2000",
        dataset_eval_names=[
            "1-1_original-pattern",
            "2-1_pred_semsim-fix_wildcard_w2v",
            "2-2_pred_semsim-fix-lemma_wildcard_cn",
            "2-3_pred_semsim-ctx_wildcard_e5_nref-10"
        ],
        eval_metrics=["f1"],
        add_random_baseline=True,
        plot_name_suffix="original_vs_semsim-fix_w2v_vs_semsim-fix-lemma_cn_vs_semsim-ctx_nref-10_e5"
    )
    plot(
        case_study=CASE_STUDY,
        dataset_name="1-2_pred_wildcard_subsample-2000",
        dataset_eval_names=[
            "1-1_original-pattern",
            "2-2_pred_semsim-fix-lemma_wildcard_cn",
            "2-3_pred_semsim-ctx_wildcard_e5_nref-10"
        ],
        eval_metrics=["precision", "recall"],
        add_random_baseline=True,
        plot_name_suffix="original_vs_semsim-fix-lemma_cn_vs_semsim-ctx_nref-10_e5"
    )
    plot(
        case_study=CASE_STUDY,
        dataset_name="1-2_pred_wildcard_subsample-2000",
        dataset_eval_names=[
            "1-1_original-pattern",
            "2-2_pred_semsim-fix-lemma_wildcard_cn",
            "2-3_pred_semsim-ctx_wildcard_e5_nref-10"
        ],
        eval_metrics=["precision", "recall", "f1"],
        plot_name_suffix="original_vs_semsim-fix-lemma_cn_vs_semsim-ctx_nref-10_e5"
    )
    plot(
        case_study=CASE_STUDY,
        dataset_name="1-2_pred_wildcard_subsample-2000",
        dataset_eval_names=[
            "2-2_pred_semsim-fix-lemma_wildcard_cn",
            "2-3_pred_semsim-ctx_wildcard_e5_nref-10"
        ],
        eval_metrics=["precision", "recall", "f1"],
        plot_name_suffix="semsim-fix-lemma_cn_vs_semsim-ctx_nref-10_e5"
    )
    plot(
        case_study=CASE_STUDY,
        dataset_name="1-2_pred_wildcard_subsample-2000",
        dataset_eval_names=[
            "2-3_pred_semsim-ctx_wildcard_e5_nref-1",
            "2-3_pred_semsim-ctx_wildcard_e5_nref-3",
            "2-3_pred_semsim-ctx_wildcard_e5_nref-10"
        ],
        eval_metrics=["f1"],
        plot_name_suffix="semsim-ctx_e5_nref-1_vs_nref-3_vs_nref-10"
    )
    # ----- SemSim CTX - E5-AT -----
    plot(
        case_study=CASE_STUDY,
        dataset_name="1-2_pred_wildcard_subsample-2000",
        dataset_eval_names=[
            "2-3_pred_semsim-ctx_wildcard_e5_nref-10",
            "2-3_pred_semsim-ctx_wildcard_e5-at_nref-10"
        ],
        eval_metrics=["precision", "recall", "f1"],
        plot_name_suffix="semsim-ctx_nref-10_e5-vs_e5-at"
    )
    # ----- SemSim CTX - GTE -----
    plot(
        case_study=CASE_STUDY,
        dataset_name="1-2_pred_wildcard_subsample-2000",
        dataset_eval_names=[
            "2-3_pred_semsim-ctx_wildcard_e5_nref-10",
            "2-3_pred_semsim-ctx_wildcard_gte_nref-10"
        ],
        eval_metrics=["precision", "recall", "f1"],
        plot_name_suffix="semsim-ctx_nref-10_e5_vs_gte"
    )
    # ----- SemSim CTX - GTE-AT -----
    plot(
        case_study=CASE_STUDY,
        dataset_name="1-2_pred_wildcard_subsample-2000",
        dataset_eval_names=[
            "1-1_original-pattern",
            "2-3_pred_semsim-ctx_wildcard_e5_nref-10",
            "2-3_pred_semsim-ctx_wildcard_gte-at_nref-10"
        ],
        eval_metrics=["precision", "recall"],
        add_random_baseline=True,
        plot_name_suffix="original_vs_semsim-ctx_nref-10_e5_vs_gte-at"
    )
    plot(
        case_study=CASE_STUDY,
        dataset_name="1-2_pred_wildcard_subsample-2000",
        dataset_eval_names=[
            "2-3_pred_semsim-ctx_wildcard_e5_nref-10",
            "2-3_pred_semsim-ctx_wildcard_e5-at_nref-10",
            "2-3_pred_semsim-ctx_wildcard_gte_nref-10",
            "2-3_pred_semsim-ctx_wildcard_gte-at_nref-10"
        ],
        eval_metrics=["f1"],
        plot_name_suffix="semsim-ctx_nref-10_e5_e5-at_gte_gte-at"
    )
