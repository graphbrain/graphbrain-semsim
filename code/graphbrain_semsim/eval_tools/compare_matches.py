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

from graphbrain.hypergraph import Hypergraph
from graphbrain_semsim import get_hgraph
from graphbrain_semsim.datasets.config import DATASET_DIR
from graphbrain_semsim.models import Hyperedge
from graphbrain_semsim.datasets.get_evaluations import get_dataset_evaluations
from graphbrain_semsim.datasets.models import DatasetEvaluation, EvaluationResult, LemmaDataset, LemmaMatch

from graphbrain_semsim.case_studies.conflicts.config import CASE_STUDY, HG_NAME
from graphbrain_semsim.utils.general import all_equal
from graphbrain_semsim.utils.file_handling import load_json

logger = logging.getLogger(__name__)


def compare_correct_matches(
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

    dataset_evaluations: dict[int, list[DatasetEvaluation]] = get_dataset_evaluations(
        dataset_eval_names, case_study, dataset_id
    )
    best_evaluation_results = get_best_evaluation_results(
        dataset_evaluations, dataset_eval_names, eval_metric, hg
    )


    # # Load dataset from file
    # dataset_file_path: Path = DATASET_DIR / f"{dataset_id}{'_recreated' if dataset_recreated else ''}.json"
    # dataset: LemmaDataset = load_json(dataset_file_path, LemmaDataset, exit_on_error=True)

    # unequal_lemma_matches_edges: dict[str, list[Hyperedge]] = get_unequal_lemma_matches_edges(dataset)
    #
    # for dataset_eval_idx, best_evaluation_result in enumerate(best_evaluation_results):
    #     logger.info(f"Checking lemma matching equality for {dataset_eval_names[dataset_eval_idx]}...")
    #     unequal_lemmas: list[str] = []
    #     for lemma, lemma_edges in unequal_lemma_matches_edges.items():
    #         if (
    #             any(edge in best_evaluation_result.matches for edge in lemma_edges) and
    #             not all(edge in best_evaluation_result.matches for edge in lemma_edges)
    #         ):
    #             unequal_lemmas.append(lemma)
    #     if unequal_lemmas:
    #         logger.info(f"Found {len(unequal_lemmas)} unequal lemmas:\n{unequal_lemmas}")
    #     else:
    #         logger.info("All lemmas have equal matching")


    # get lemmmas with the highest difference in evalluation score for the two dataset evaluations
    # for



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


def get_best_evaluation_results(
    dataset_evaluations: dict[int, list[DatasetEvaluation]],
    dataset_eval_names: list[str],
    eval_metric: str,
    hg: Hypergraph
) -> dict[str, tuple[EvaluationResult, float | None, list[Hyperedge] | None]]:
    best_evaluation_results: dict[str, tuple[EvaluationResult, float | None, list[Hyperedge] | None]] = {}
    for dataset_eval_idx, dataset_sub_evaluations in dataset_evaluations.items():
        dataset_eval_name: str = dataset_eval_names[dataset_eval_idx]
        best_evaluation_result, best_threshold, best_ref_edges = get_best_evaluation_result(
            dataset_sub_evaluations, eval_metric
        )
        best_evaluation_results[dataset_eval_name] = best_evaluation_result, best_threshold, best_ref_edges

        logger.info(f"Best evaluation result for dataset evaluation {dataset_eval_name}:")
        logger.info(f"--> evaluation metric {eval_metric}: {getattr(best_evaluation_result, eval_metric)}")
        if best_threshold is not None:
            logger.info(f"--> best threshold: {best_threshold}")
        if best_ref_edges is not None:
            logger.info("--> best ref. edges:" + '\n'.join([hg.text(edge) for edge in best_ref_edges]))

    return best_evaluation_results


def get_best_evaluation_result(
    dataset_evaluations: list[DatasetEvaluation],
    eval_metric: str,
) -> tuple[EvaluationResult, float | None, list[Hyperedge] | None]:
    if len(dataset_evaluations) == 1:
        best_result, best_threshold = get_best_evaluation_result_from_single(dataset_evaluations[0], eval_metric)
        return best_result, best_threshold, None
    else:
        results_and_thresholds: list[tuple[EvaluationResult, float]] = [
            get_best_evaluation_result_from_single(dataset_evaluation, eval_metric)
            for dataset_evaluation in dataset_evaluations
        ]
        best_result_idx: int = max(enumerate(results_and_thresholds), key=lambda t: getattr(t[1][0], eval_metric))[0]
        best_evaluation: DatasetEvaluation = dataset_evaluations[best_result_idx]
        best_result, best_threshold = results_and_thresholds[best_result_idx]
        return best_result, best_threshold, best_evaluation.ref_edges


def get_best_evaluation_result_from_single(
    dataset_evaluation: DatasetEvaluation,
    eval_metric: str,
) -> tuple[EvaluationResult, float | None]:
    if dataset_evaluation.mean_symbolic_eval_results:
        return dataset_evaluation.mean_symbolic_eval_results, None

    if dataset_evaluation.mean_semsim_eval_results:
        best_threshold, best_result = max(
            dataset_evaluation.semsim_eval_results.items(),
            key=lambda threshold_result: getattr(threshold_result[1], eval_metric)
        )
        return best_result, best_threshold


# def get_unequal_lemma_matches_edges(dataset: LemmaDataset) -> dict[str, list[Hyperedge]]:
#     return {
#         lemma: [lemma_match_.match.edge for lemma_match_ in lemma_matches_]
#         for lemma, lemma_matches_ in dataset.lemma_matches.items()
#         if not all_equal([lemma_match_.label for lemma_match_ in lemma_matches_])
#     }


compare_correct_matches(
    dataset_name="1-1_wildcard_preds_subsample-2000",
    dataset_recreated=True,
    dataset_eval_names=[
            "2-2_preds_semsim-fix-lemma_wildcard",
            "2-3_preds_semsim-ctx_wildcard_nref-10"
        ],
    eval_metric="f1",
)
