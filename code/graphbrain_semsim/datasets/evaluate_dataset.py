import random
from pathlib import Path

from tqdm import tqdm

from graphbrain.hypergraph import Hyperedge, Hypergraph, hedge
from graphbrain.patterns.semsim.processing import match_semsim_instances
from graphbrain.semsim import init_matcher, SemSimConfig, SemSimType
from graphbrain_semsim import logger, get_hgraph, RNG_SEED
from graphbrain_semsim.case_studies.models import PatternEvaluationConfig, PatternEvaluationRun
from graphbrain_semsim.case_studies.evaluate_pattern import evaluate_pattern
from graphbrain_semsim.datasets.config import DATA_LABELS, DATASET_DIR, DATASET_EVAL_DIR
from graphbrain_semsim.datasets.models import LemmaDataset, EvaluationScore, DatasetEvaluation
from graphbrain_semsim.eval_tools.utils.result_data import get_pattern_eval_config, get_pattern_eval_run
from graphbrain_semsim.utils.general import load_json, frange, save_json

from graphbrain_semsim.case_studies.conflicts.pattern_configs import (
    PATTERN_CONFIGS, SUB_PATTERN_WORDS, ConflictsSubPattern
)

random.seed(RNG_SEED)

_HG_STORE: dict[str, Hypergraph] = {}


EVALUATION_FILE_SUFFIX: str = "evaluation"


def evaluate_dataset_for_pattern(
        dataset_name: str,
        pattern_config_name: str,
        pattern_configs: list[PatternEvaluationConfig],
        semsim_threshold_range: list[float] = None,
        semsim_configs: dict[SemSimType, SemSimConfig] = None,
        ref_words: list[str] = None,
        n_ref_edges: int = None,
        sample_mod: int = None,
        override: bool = False
) -> DatasetEvaluation:
    logger.info("-" * 80)
    logger.info(f"Evaluating dataset '{dataset_name}' for pattern '{pattern_config_name}'...")

    # Load dataset from file
    dataset: LemmaDataset = load_json(DATASET_DIR / f"{dataset_name}.json", LemmaDataset, exit_on_error=True)
    dataset_positives, dataset_negatives = get_dataset_positives_and_negatives(dataset)
    log_dataset_statistics(dataset, dataset_positives, dataset_negatives)

    # Get pattern evaluation config and run
    pattern_eval_config, pattern_eval_run = get_pattern_eval(
        pattern_configs, pattern_config_name, dataset, override=override
    )

    ref_edges: list[Hyperedge] | None = None
    if n_ref_edges:
        ref_edges: list[Hyperedge] = sample_ref_edges(dataset_positives, n_ref_edges, sample_mod)

    dataset_evaluation: DatasetEvaluation = DatasetEvaluation(
        dataset_name=dataset_name,
        pattern_eval_config_id=pattern_eval_config.id,
        num_samples=dataset.n_samples,
        num_positive=len(dataset_positives),
        num_negative=len(dataset_negatives),
        semsim_configs=semsim_configs,
        ref_words=ref_words,
        ref_edges=ref_edges,
    )

    process_eval_scores(
        dataset_evaluation,
        pattern_eval_run,
        semsim_threshold_range,
        pattern_eval_config.hypergraph,
        dataset_positives,
        dataset_negatives,
    )

    dataset_evaluation_file_path: Path = get_dataset_evaluation_file_path(
        dataset_name, pattern_eval_config.id, n_ref_edges, sample_mod
    )

    save_json(dataset_evaluation, dataset_evaluation_file_path)
    return dataset_evaluation


def get_dataset_positives_and_negatives(dataset: LemmaDataset) -> tuple[list[Hyperedge], list[Hyperedge]]:
    dataset_positives: list[Hyperedge] = [
        lemma_match.match.edge for lemma_match in dataset.all_lemma_matches
        if lemma_match.label == DATA_LABELS['positive']
    ]
    dataset_negatives: list[Hyperedge] = [
        lemma_match.match.edge for lemma_match in dataset.all_lemma_matches
        if lemma_match.label == DATA_LABELS['negative']
    ]

    assert len(dataset_positives) + len(dataset_negatives) == dataset.n_samples, (
        f"Number of positive and negative samples does not match total number of samples in dataset '{dataset.name}'"
    )

    return dataset_positives, dataset_negatives


def log_dataset_statistics(dataset: LemmaDataset, positives: list[Hyperedge], negatives: list[Hyperedge]):
    num_positives: int = len(positives)
    num_negatives: int = len(negatives)

    logger.info(
        f"Dataset statistics for '{dataset.name}':\n"
        f"  - Total number of samples: {dataset.n_samples}\n"
        f"  - Number of positives: {num_positives} ({num_positives / dataset.n_samples * 100:.2f} %)\n"
        f"  - Number of negatives: {num_negatives} ({num_negatives / dataset.n_samples * 100:.2f} %)\n"
    )


def get_pattern_eval(
        pattern_configs: list[PatternEvaluationConfig],
        pattern_config_name: str,
        dataset: LemmaDataset,
        override: bool = False
) -> tuple[PatternEvaluationConfig, PatternEvaluationRun]:
    # Run scenario against dataset
    pattern_eval_config: PatternEvaluationConfig = get_pattern_eval_config(
        pattern_configs, case_study=dataset.case_study, pattern_config_name=pattern_config_name
    )
    assert pattern_eval_config.hypergraph == dataset.hg_name, (
        f"Hypergraph of pattern evaluation config '{pattern_eval_config.name}' "
        f"does not match hypergraph of dataset '{dataset.name}'"
    )

    # Try to get the evaluation run file if override is not enabled
    logger.info(f"### Override pattern eval run: {override}")
    pattern_eval_run: PatternEvaluationRun | None = None
    if not override:
        pattern_eval_run: PatternEvaluationRun | None = get_pattern_eval_run(
            pattern_config_id=pattern_eval_config.id, dataset_name=dataset.name
        )

    # Run the pattern evaluation if not loaded from file
    if not pattern_eval_run:
        pattern_eval_runs: list[PatternEvaluationRun] = evaluate_pattern(
            pattern_eval_config, dataset=dataset, override=override
        )
        assert len(pattern_eval_runs) == 1, "Only one eval run should be returned"
        pattern_eval_run: PatternEvaluationRun = pattern_eval_runs[0]

    return pattern_eval_config, pattern_eval_run


def process_eval_scores(
        dataset_evaluation: DatasetEvaluation,
        pattern_eval_run: PatternEvaluationRun,
        semsim_threshold_range: list[float],
        hg_name: str,
        dataset_positives: list[Hyperedge],
        dataset_negatives: list[Hyperedge],
) -> DatasetEvaluation:
    if semsim_threshold_range:
        assert dataset_evaluation.ref_words or dataset_evaluation.ref_edges, (
            "Either reference words or reference edges must be given when using SemSim"
        )
        dataset_evaluation.semsim_eval_scores = get_semsim_eval_scores(
            pattern_eval_run,
            dataset_evaluation.semsim_configs,
            semsim_threshold_range,
            dataset_evaluation.ref_words,
            dataset_evaluation.ref_edges,
            hg_name,
            dataset_positives,
            dataset_negatives,
        )
    else:
        dataset_evaluation.symbolic_eval_score = get_symbolic_eval_score(
            pattern_eval_run,
            dataset_positives,
            dataset_negatives,
        )
    return dataset_evaluation


def get_symbolic_eval_score(
        pattern_eval_run: PatternEvaluationRun,
        dataset_positives: list[Hyperedge],
        dataset_negatives: list[Hyperedge],
) -> EvaluationScore:
    logger.info("Computing symbolic evaluation score...")
    eval_run_positives: list[Hyperedge] = [
        match.edge for match in pattern_eval_run.matches
    ]
    eval_run_negatives: list[Hyperedge] = [
        edge for edge in dataset_positives + dataset_negatives
        if edge not in eval_run_positives
    ]

    logger.info("Done computing symbolic evaluation score!")
    return compute_eval_score(
        dataset_positives,
        dataset_negatives,
        eval_run_positives,
        eval_run_negatives
    )


def get_semsim_eval_scores(
        pattern_eval_run: PatternEvaluationRun,
        semsim_configs: dict[SemSimType, SemSimConfig],
        semsim_threshold_range: list[float],
        ref_words: list[str],
        ref_edges: list[Hyperedge],
        hg_name: str,
        dataset_positives: list[Hyperedge],
        dataset_negatives: list[Hyperedge],
) -> dict[float, EvaluationScore]:
    logger.info("Computing SemSim evaluation scores...")

    if hg_name not in _HG_STORE:
        _HG_STORE[hg_name] = get_hgraph(hg_name)
    hg: Hypergraph = _HG_STORE[hg_name]

    # Initialize the semsim matcher if configs given
    if semsim_configs:
        for matcher_type, semsim_config in semsim_configs.items():
            init_matcher(matcher_type, semsim_config)

    eval_scores: dict[float, EvaluationScore] = {}
    for st_idx, semsim_threshold in enumerate(semsim_threshold_range):
        logger.info(f"Getting matches for threshold [{st_idx + 1}/{len(semsim_threshold_range)}]: {semsim_threshold}")

        eval_run_positives: list[Hyperedge] = get_post_semsim_match_edges(
            pattern_eval_run, semsim_threshold, ref_words, ref_edges, hg
        )
        eval_run_negatives: list[Hyperedge] = [
            match.edge for match in pattern_eval_run.matches
            if match.edge not in eval_run_positives
        ]

        # Compute evaluation score
        eval_scores[semsim_threshold] = compute_eval_score(
            dataset_positives,
            dataset_negatives,
            eval_run_positives,
            eval_run_negatives
        )

    return eval_scores


def get_post_semsim_match_edges(
        pattern_eval_run: PatternEvaluationRun,
        threshold: float,
        ref_words: list[str],
        ref_edges: list[Hyperedge],
        hg: Hypergraph
) -> list[Hyperedge]:
    post_semsim_match_edges: list[Hyperedge] = []
    for match in tqdm(pattern_eval_run.matches):
        # either match has no semsim instances
        # or they need to be matched now
        if not match.semsim_instances or match_semsim_instances(
            semsim_instances=match.semsim_instances,
            pattern=hedge(pattern_eval_run.pattern),
            edge=match.edge,
            hg=hg,
            threshold=threshold,
            ref_words=ref_words,
            ref_edges=ref_edges,
        ):
            post_semsim_match_edges.append(match.edge)
    return post_semsim_match_edges


def sample_ref_edges(
        dataset_positives: list[Hyperedge], 
        n_ref_edges: int,
        sample_mod: int = None
) -> list[Hyperedge]:
    if sample_mod:
        random.seed(RNG_SEED + sample_mod)
    return list(random.sample(dataset_positives, k=n_ref_edges))


def compute_eval_score(
        dataset_positives: list[Hyperedge],
        dataset_negatives: list[Hyperedge],
        eval_run_positives: list[Hyperedge],
        eval_run_negatives: list[Hyperedge],
) -> EvaluationScore:
    # Compute precision, recall, and F1 score
    true_positives: int = len(set(dataset_positives) & set(eval_run_positives))
    false_positives: int = len(set(dataset_negatives) & set(eval_run_positives))
    false_negatives: int = len(set(dataset_positives) & set(eval_run_negatives))

    precision: float = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0.0
    recall: float = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0.0
    f1: float = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

    return EvaluationScore(
        precision=precision,
        recall=recall,
        f1=f1,
    )


def get_dataset_evaluation_file_path(
    dataset_name: str,
    pattern_eval_config_id: str,
    n_ref_edges: int = None,
    sample_mod: int = None,
):
    n_ref_edges_descriptor: str | None = f"nref-{n_ref_edges}" if n_ref_edges else None
    sample_mod_descriptor: str | None = f"smod-{sample_mod}" if sample_mod else None

    dataset_evaluation_descriptor: str = f"{dataset_name}_{EVALUATION_FILE_SUFFIX}_{pattern_eval_config_id}"

    dataset_evaluation_file_stem: str = dataset_evaluation_descriptor
    if n_ref_edges:
        dataset_evaluation_file_stem += f"_{n_ref_edges_descriptor}"
    if n_ref_edges and sample_mod:
        dataset_evaluation_file_stem += f"_{sample_mod_descriptor}"

    dataset_evaluation_file_dir: Path = DATASET_EVAL_DIR / dataset_evaluation_descriptor
    if n_ref_edges:
        dataset_evaluation_file_dir /= n_ref_edges_descriptor

    return dataset_evaluation_file_dir / f"{dataset_evaluation_file_stem}.json"


if __name__ == "__main__":
    # evaluate_dataset_for_pattern(
    #     dataset_name="dataset_conflicts_1-1_wildcard_preds_subsample-2000_recreated",
    #     pattern_config_name="1-1_original-pattern",
    #     pattern_configs=PATTERN_CONFIGS,
    #     override=False
    # )
    # evaluate_dataset_for_pattern(
    #     dataset_name="dataset_conflicts_1-1_wildcard_preds_subsample-2000_recreated",
    #     pattern_config_name="2-1_preds_semsim-fix_wildcard",
    #     pattern_configs=PATTERN_CONFIGS,
    #     semsim_threshold_range=frange(0.0, 1.0, 0.05),
    #     ref_words=SUB_PATTERN_WORDS[ConflictsSubPattern.PREDS],
    #     override=True
    # )
    # evaluate_dataset_for_pattern(
    #     dataset_name="dataset_conflicts_1-1_wildcard_preds_subsample-2000_recreated",
    #     pattern_config_name="2-2_preds_semsim-fix-lemma_wildcard",
    #     pattern_configs=PATTERN_CONFIGS,
    #     semsim_threshold_range=frange(0.0, 1.0, 0.05),
    #     ref_words=SUB_PATTERN_WORDS[ConflictsSubPattern.PREDS],
    #     override=True
    # )

    for sample_mod_ in range(5):
        for n_ref_edges_ in [1, 5, 10]:
            evaluate_dataset_for_pattern(
                dataset_name="dataset_conflicts_1-1_wildcard_preds_subsample-2000_recreated",
                pattern_config_name="2-3_preds_semsim-ctx_wildcard",
                pattern_configs=PATTERN_CONFIGS,
                semsim_threshold_range=frange(0.0, 1.0, 0.05),
                n_ref_edges=n_ref_edges_,
                sample_mod=sample_mod_,
            )

