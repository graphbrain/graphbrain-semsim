"""
Given two dataset evaluation names and an evaluation metric, compare the matches and show the differences.
1. Load the two dataset evaluations. If one of them refers to a directory, load all the evaluations in that directory.
2. For each dataset evaluation, get the evaluation result which has the highest score in the given evaluation metric.
  2.1 For symbolic matching there is only one evaluation result.
  2.2 For semsim-fix, iterate over the similarity thresholds and get the evaluation result with the highest score.
  2.2 For semsim-ctx, iterate over the different samples of ref. edges and over the similarity thresholds
  and get the evaluation result with the highest score.
3. Get the matches of the two evaluation results and compare them.
"""
import logging
from pathlib import Path

from graphbrain.hypergraph import Hypergraph, Hyperedge
from graphbrain_semsim import get_hgraph
from graphbrain_semsim.datasets.config import DATASET_DIR, CONFLICTS_ANNOTATION_LABELS
from graphbrain_semsim.datasets.get_evaluations import get_dataset_evaluations
from graphbrain_semsim.datasets.models import DatasetEvaluation, EvaluationResult, LemmaDataset

from graphbrain_semsim.case_studies.conflicts.config import CASE_STUDY, HG_NAME
from graphbrain_semsim.eval_tools.result_data.dataset_evals import get_best_evaluations_and_results_and_thresholds
from graphbrain_semsim.utils.file_handling import load_json

logger = logging.getLogger(__name__)

REVERSE_CONFLICTS_ANNOTATION_LABELS: dict[int, str] = {
    label: name for name, label in CONFLICTS_ANNOTATION_LABELS.items()
}


def compare_dataset_evaluations(
    dataset_name: str,
    dataset_recreated: bool,
    dataset_eval_names: list[str],
    eval_metric: str,
    hg_name: str = HG_NAME,
    case_study: str = CASE_STUDY,
):
    assert len(dataset_eval_names) == 2, "Two dataset evaluation names must be provided"

    hg: Hypergraph = get_hgraph(hg_name)

    dataset_id: str = f"dataset_{case_study}_{dataset_name}"

    # Load dataset from file
    dataset_file_path: Path = DATASET_DIR / f"{dataset_id}{'_recreated' if dataset_recreated else ''}.json"
    dataset: LemmaDataset = load_json(dataset_file_path, LemmaDataset, exit_on_error=True)

    dataset_evaluations: list[list[DatasetEvaluation]] = get_dataset_evaluations(
        dataset_eval_names, case_study, dataset_id
    )
    best_evaluations_results_thresholds: dict[str, tuple[DatasetEvaluation, EvaluationResult, float | None]] = (
        get_best_evaluations_and_results_and_thresholds(
            dataset_evaluations, dataset_eval_names, eval_metric, hg
        )
    )

    compare_lemma_results_by_eval_metric_diff(
        best_evaluations_results_thresholds, dataset_eval_names, dataset, eval_metric, hg
    )

    compare_matches(
        [best_result for _, best_result, _ in best_evaluations_results_thresholds.values()], dataset_eval_names, hg
    )


def compare_lemma_results_by_eval_metric_diff(
    best_evaluations_results_thresholds: dict[str, tuple[DatasetEvaluation, EvaluationResult, float | None]],
    dataset_eval_names: list[str],
    dataset: LemmaDataset,
    rank_metric: str,
    hg: Hypergraph
):
    rank_metric: str = "f1"

    min_samples: int = 5
    lemma_compare_metric: str = "precision"

    # get lemmas with the highest difference in evaluation score for the two dataset evaluations
    best_evaluation_1, _, best_threshold_1 = best_evaluations_results_thresholds[dataset_eval_names[0]]
    best_evaluation_2, _, best_threshold_2 = best_evaluations_results_thresholds[dataset_eval_names[1]]

    best_lemma_results_1: dict[str, EvaluationResult] = get_best_lemma_results(best_evaluation_1, best_threshold_1)
    best_lemma_results_2: dict[str, EvaluationResult] = get_best_lemma_results(best_evaluation_2, best_threshold_2)
    assert best_lemma_results_1.keys() == best_lemma_results_2.keys(), (
        "The two dataset evaluations must have the same lemmas"
    )

    for dataset_eval_idx, best_lemma_results in enumerate([best_lemma_results_1, best_lemma_results_2]):
        best_lemma_results_sorted: list[tuple[str, EvaluationResult]] = list(sorted(
            best_lemma_results.items(), key=lambda t: getattr(t[1], rank_metric), reverse=True
        ))
        best_lemma_results_sorted_filtered: list[tuple[str, EvaluationResult]] = [
            (lemma, result) for lemma, result in best_lemma_results_sorted
            if len(dataset.lemma_matches[lemma]) >= min_samples
        ]

        logger.info(
            f"Top 10 lemmas regarding {rank_metric} with n_samples >= {min_samples} "   
            f"for {dataset_eval_names[dataset_eval_idx]}:\n" +
            "\n".join(
                f"{lemma}: {getattr(result, rank_metric):.2f} (n={len(dataset.lemma_matches[lemma])})"
                for lemma, result in best_lemma_results_sorted_filtered[:10]
            )
        )

    lemma_results_scores_and_diffs: list[tuple[str, tuple[float, float, float]]] = []
    for lemma in dataset.lemma_matches.keys():
        best_lemma_result_1: EvaluationResult = best_lemma_results_1[lemma]
        best_lemma_result_2: EvaluationResult = best_lemma_results_2[lemma]

        if best_lemma_result_1.recall > 0 and best_lemma_result_2.recall > 0:
            best_lemma_result_score_1: float = getattr(best_lemma_result_1, lemma_compare_metric)
            best_lemma_result_score_2: float = getattr(best_lemma_result_2, lemma_compare_metric)
            lemma_results_scores_and_diffs.append((
                lemma, (
                    best_lemma_result_score_1,
                    best_lemma_result_score_2,
                    abs(best_lemma_result_score_1 - best_lemma_result_score_2)
                )
            ))

    lemma_results_scores_and_diffs_sorted: list[tuple[str, tuple[float, float, float]]] = list(sorted(
        lemma_results_scores_and_diffs, key=lambda t: t[1][2], reverse=True
    ))

    # lemmas with at least n samples in the dataset
    lemma_results_scores_and_diffs_sorted_filtered: list[tuple[str, tuple[float, float, float]]] = [
        (lemma, scores_and_diff) for lemma, scores_and_diff in lemma_results_scores_and_diffs_sorted
        if len(dataset.lemma_matches[lemma]) >= min_samples
    ]

    logger.info(
        f"Top 10 lemmas regarding highest difference in {lemma_compare_metric}\n"
        f"limited to lemmas with recall > 0 and n_samples >= {min_samples}\n"
        f"for {dataset_eval_names[0]} and {dataset_eval_names[1]}:\n" +
        "\n".join(
            f"{lemma}: {diff:.2f} ({score_1:.2f} - {score_2:.2f}) (n={len(dataset.lemma_matches[lemma])})"
            for lemma, (score_1, score_2, diff) in lemma_results_scores_and_diffs_sorted_filtered[:10]
        )
    )
    logger.info(
        f"Top 10 lemmas regarding lowest difference in {lemma_compare_metric}\n"
        f"limited to lemmas with recall > 0 and n_samples >= {min_samples}\n"
        f"for {dataset_eval_names[0]} and {dataset_eval_names[1]}:\n" +
        "\n".join(
            f"{lemma}: {diff:.2f} ({score_1:.2f} - {score_2:.2f}) (n={len(dataset.lemma_matches[lemma])})"
            for lemma, (score_1, score_2, diff) in reversed(lemma_results_scores_and_diffs_sorted_filtered[-10:])
        )
    )

    logger.info(
        f"Edges for top 5 lemmas with highest difference in {lemma_compare_metric}\n"
        f"limited to lemmas with recall > 0 and n_samples >= {min_samples}\n"
        f"for {dataset_eval_names[0]} and {dataset_eval_names[1]}:\n" +
        "\n".join(
            f"{lemma}:\n" +
            "\n".join(
                f"  {hg.text(lemma_match.match.edge)} "
                f"(ground_truth: {REVERSE_CONFLICTS_ANNOTATION_LABELS[lemma_match.label]}) "
                f"(eval_1: {get_eval_edge_label_name(lemma_match.match.edge, best_evaluation_1, best_threshold_1)}) "
                f"(eval_2: {get_eval_edge_label_name(lemma_match.match.edge, best_evaluation_2, best_threshold_2)})"
                for lemma_match in dataset.lemma_matches[lemma]
            )
            for lemma, _ in lemma_results_scores_and_diffs_sorted_filtered[:5]
        )
    )
                

def get_eval_edge_label_name(edge: Hyperedge, dataset_evaluation: DatasetEvaluation, semsim_threshold: float) -> str:
    evaluation_matches_edges: list[Hyperedge] = (
        dataset_evaluation.symbolic_eval_results.matches if dataset_evaluation.symbolic_eval_result
        else dataset_evaluation.semsim_eval_results[semsim_threshold].matches
    )
    if edge in evaluation_matches_edges:
        return REVERSE_CONFLICTS_ANNOTATION_LABELS[CONFLICTS_ANNOTATION_LABELS['conflict']]
    return REVERSE_CONFLICTS_ANNOTATION_LABELS[CONFLICTS_ANNOTATION_LABELS['no_conflict']]


def get_best_lemma_results(dataset_evaluation: DatasetEvaluation, best_threshold: float) -> dict[str, EvaluationResult]:
    if dataset_evaluation.symbolic_eval_result:
        return dataset_evaluation.lemma_symbolic_eval_results

    if dataset_evaluation.semsim_eval_results:
        return {
            lemma: lemma_eval_results[best_threshold]
            for lemma, lemma_eval_results in dataset_evaluation.lemma_semsim_eval_results.items()
        }


def compare_matches(
    best_evaluation_results: list[EvaluationResult],
    dataset_eval_names: list[str],
    hg: Hypergraph
):
    # compare matches of the two evaluation results
    correct1_set = set(best_evaluation_results[0].correct)
    correct2_set = set(best_evaluation_results[1].correct)
    correct1_only = correct1_set - correct2_set
    correct2_only = correct2_set - correct1_set

    logger.info(f"Number of correct matches in {dataset_eval_names[0]}: {len(correct1_set)}")
    logger.info(f"Number of correct matches in {dataset_eval_names[1]}: {len(correct2_set)}")
    logger.info(f"Number of correct matches in both: {len(correct1_set & correct2_set)}")
    logger.info(f"Number of correct matches only in {dataset_eval_names[0]}: {len(correct1_only)}")
    logger.info(f"Number of correct matches only in {dataset_eval_names[1]}: {len(correct2_only)}")

    if correct1_only:
        logger.info("-----")
    logger.info(f"Matches in {dataset_eval_names[0]} but not in {dataset_eval_names[1]}:\n-----")
    logger.info("\n" + "\n".join(f"{hg.text(match)}" for match in correct1_only))

    if correct2_only:
        logger.info("-----")
    logger.info(f"Matches in {dataset_eval_names[1]} but not in {dataset_eval_names[0]}:\n-----")
    logger.info("\n" + "\n".join(f"{hg.text(match)}" for match in correct2_only))


compare_dataset_evaluations(
    dataset_name="1-2_pred_wildcard_subsample-2000",
    dataset_recreated=True,
    dataset_eval_names=[
            "2-2_pred_semsim-fix-lemma_wildcard",
            "2-3_pred_semsim-ctx_wildcard_nref-10"
        ],
    eval_metric="f1",
)
