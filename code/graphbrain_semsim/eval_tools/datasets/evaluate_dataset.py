from pathlib import Path
from typing import Optional

from pydantic import BaseModel

from graphbrain.semsim import SemSimType
from graphbrain.semsim.matcher.fixed_matcher import FixedEmbeddingMatcher
from graphbrain_semsim import logger
from graphbrain_semsim.conflicts_case_study.config import SUB_PATTERN_WORDS, ConflictsSubPattern
from graphbrain_semsim.conflicts_case_study.models import EvaluationScenario
from graphbrain_semsim.conflicts_case_study.scenario_configs import EVAL_SCENARIOS
from graphbrain_semsim.eval_tools.datasets.config import DATA_DIR
from graphbrain_semsim.eval_tools.datasets.dataset_models import LemmaDataset, LemmaMatch
from graphbrain_semsim.eval_tools.utils.result_data import get_eval_scenario
from graphbrain_semsim.utils.general import load_json, frange, save_json


EVALUATION_FILE_SUFFIX: str = "evaluation"


class EvaluationScore(BaseModel):
    matches: list[LemmaMatch]
    precision: float
    recall: float
    f1: float


class DatasetEvaluation(BaseModel):
    dataset_name: str
    ref_words: list[str]
    num_samples: int
    num_conflicts: int
    num_no_conflicts: int
    num_unlabeled: int
    lemma_match_eval_score: Optional[EvaluationScore] = None
    semsim_match_eval_scores: Optional[dict[float, EvaluationScore]] = None
    contradicting_matches: Optional[dict[str, tuple[str, float]]] = None


def evaluate_dataset(dataset_name: str) -> DatasetEvaluation:
    # Load dataset from file
    dataset: LemmaDataset = load_json(DATA_DIR / f"{dataset_name}.json", LemmaDataset)

    # Log dataset statistics
    all_lemma_matches: list[LemmaMatch] = [
        lemma_match for lemma_matches in dataset.lemma_matches.values() for lemma_match in lemma_matches
    ]
    num_conflicts: int = len([
        lemma_match for lemma_match in all_lemma_matches if lemma_match.label == 1
    ])
    num_no_conflicts: int = len([
        lemma_match for lemma_match in all_lemma_matches if lemma_match.label == 2
    ])
    num_unlabeled: int = len([
        lemma_match for lemma_match in all_lemma_matches if lemma_match.label is None
    ])

    logger.info(
        f"Dataset statistics for '{dataset_name}':\n"
        f"  - Total number of samples: {dataset.n_samples}\n"
        f"  - Number of conflicts: {num_conflicts} ({num_conflicts/dataset.n_samples * 100:.2f} %)\n"
        f"  - Number of no conflicts: {num_no_conflicts} ({num_no_conflicts/dataset.n_samples * 100:.2f} %)\n"
        f"  - Number of unlabeled: {num_unlabeled} ({num_unlabeled/dataset.n_samples * 100:.2f} %)\n"
    )

    ref_words: list[str] = SUB_PATTERN_WORDS[ConflictsSubPattern.PREDS]

    dataset_evaluation: DatasetEvaluation = DatasetEvaluation(
        dataset_name=dataset_name,
        num_samples=dataset.n_samples,
        num_conflicts=num_conflicts,
        num_no_conflicts=num_no_conflicts,
        num_unlabeled=num_unlabeled,
        ref_words=ref_words,
    )

    # Given a set of reference words, evaluate the dataset against them
    # Compute precision, recall, and F1 score (for different thresholds)

    eval_scenario: EvaluationScenario = get_eval_scenario(EVAL_SCENARIOS, scenario_id=dataset.scenario_id)
    semsim_matcher: FixedEmbeddingMatcher = FixedEmbeddingMatcher(eval_scenario.semsim_configs[SemSimType.FIX])

    dataset_evaluation.lemma_match_eval_score = compute_eval_score(
        dataset_evaluation=dataset_evaluation, all_lemma_matches=all_lemma_matches
    )
    dataset_evaluation.semsim_match_eval_scores = {
        threshold: compute_eval_score(
            dataset_evaluation=dataset_evaluation, all_lemma_matches=all_lemma_matches,
            semsim_matcher=semsim_matcher, semsim_threshold=threshold
        ) for threshold in frange(0, 1, 0.01)
    }

    if dataset_evaluation.contradicting_matches:
        logger.info(
            f"Contradicting matches (lemma does match, but semsim does not):\n" + "\n".join([
                f"  - {word}: {lemma_match}" for word, lemma_match in dataset_evaluation.contradicting_matches.items()
            ])
        )

    dataset_evaluation_file_path: Path = DATA_DIR / f"{dataset_name}_{EVALUATION_FILE_SUFFIX}.json"
    logger.info(f"Saving dataset evaluation to '{dataset_evaluation_file_path}'...")
    save_json(dataset_evaluation, dataset_evaluation_file_path)
    return dataset_evaluation


def compute_eval_score(
        dataset_evaluation: DatasetEvaluation,
        all_lemma_matches: list[LemmaMatch],
        semsim_matcher: FixedEmbeddingMatcher = None,
        semsim_threshold: float = None,
) -> EvaluationScore:
    if semsim_matcher:
        assert semsim_threshold is not None, "SemSim threshold must be provided if SemSim matcher is provided"
        # matches: list[LemmaMatch] = [
        #     match for match in all_lemma_matches if semsim_matcher.similar(
        #         cand_word=match.word, ref_words=dataset_evaluation.ref_words, threshold=semsim_threshold
        #     )
        # ]
        # implemented as loop for debugging purposes
        matches: list[LemmaMatch] = []
        for match in all_lemma_matches:
            if similar := semsim_matcher.similar(
                    cand_word=match.word, ref_words=dataset_evaluation.ref_words, threshold=semsim_threshold
            ):
                matches.append(match)
            if not similar and match.lemma in dataset_evaluation.ref_words:
                if not dataset_evaluation.contradicting_matches:
                    dataset_evaluation.contradicting_matches = {}
                if match.word not in dataset_evaluation.contradicting_matches:
                    dataset_evaluation.contradicting_matches[match.word] = (match.lemma, semsim_threshold)
    else:
        matches: list[LemmaMatch] = [
            match for match in all_lemma_matches if match.lemma in dataset_evaluation.ref_words
        ]

    num_matches: int = len(matches)
    num_conflict_matches: int = len([match for match in matches if match.label == 1])

    precision: float = num_conflict_matches / num_matches
    recall: float = num_conflict_matches / dataset_evaluation.num_conflicts
    f1: float = 2 * (precision * recall) / (precision + recall)

    return EvaluationScore(
        matches=matches,
        precision=precision,
        recall=recall,
        f1=f1,
    )


if __name__ == "__main__":
    evaluate_dataset("dataset_conflicts_1-1_wildcard_preds_subsample-2000_recreated")
