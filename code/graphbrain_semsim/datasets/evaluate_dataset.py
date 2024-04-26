import random
from pathlib import Path

from tqdm import tqdm

from graphbrain import hedge
from graphbrain.hypergraph import Hypergraph
from graphbrain.patterns.semsim.processing import match_semsim_instances
from graphbrain.semsim import init_matcher, SemSimConfig, SemSimType
from graphbrain_semsim import logger, get_hgraph, RNG_SEED
from graphbrain_semsim.models import Hyperedge
from graphbrain_semsim.case_studies.models import PatternEvaluationConfig, PatternEvaluationRun, PatternMatch
from graphbrain_semsim.case_studies.evaluate_pattern import evaluate_pattern
from graphbrain_semsim.datasets.config import DATA_LABELS, DATASET_DIR, DATASET_EVAL_DIR
from graphbrain_semsim.datasets.models import LemmaDataset, EvaluationResult, DatasetEvaluation, LemmaMatch
from graphbrain_semsim.eval_tools.result_data.pattern_eval_runs import get_pattern_eval_config, get_pattern_eval_run
from graphbrain_semsim.utils.file_handling import save_json, load_json


EVALUATION_FILE_SUFFIX: str = "evaluation"


def evaluate_dataset_for_pattern(
        dataset_id: str,
        pattern_config_name: str,
        pattern_configs: list[PatternEvaluationConfig],
        semsim_threshold_range: list[float] = None,
        semsim_configs_name: str = None,
        semsim_eval_configs: dict[str, dict[SemSimType, SemSimConfig]] = None,
        ref_words: list[str] = None,
        n_ref_edges: int = None,
        sample_mod: int = None,
        override: bool = False,
        only_count_matches: bool = False,
        ref_edges_dataset_id: str = None,
) -> DatasetEvaluation:
    logger.info("-" * 80)
    logger.info(f"Evaluating dataset '{dataset_id}' for pattern '{pattern_config_name}'...")

    # Load dataset from file
    dataset: LemmaDataset = load_json(DATASET_DIR / f"{dataset_id}.json", LemmaDataset, exit_on_error=True)

    ref_edges_dataset = load_json(
        DATASET_DIR / f"{ref_edges_dataset_id}.json", LemmaDataset, exit_on_error=True
    ) if ref_edges_dataset_id else dataset

    dataset_positives, dataset_negatives = None, None
    if not only_count_matches:
        dataset_positives, dataset_negatives = get_positives_and_negatives(dataset.all_lemma_matches)
        log_dataset_statistics(dataset, dataset_positives, dataset_negatives)

    dataset_positives_ref, _ = get_positives_and_negatives(ref_edges_dataset.all_lemma_matches)

    # Get pattern evaluation parse_config and run
    pattern_eval_config, pattern_eval_run = get_pattern_eval(
        pattern_configs, pattern_config_name, dataset, override=override
    )

    ref_edges: list[Hyperedge] | None = get_ref_edges(dataset_positives_ref, n_ref_edges, sample_mod)
    if not ref_edges_dataset_id and not only_count_matches:
        dataset_positives = filter_dataset_positives(dataset_positives, ref_edges)

    semsim_configs: dict[SemSimType, SemSimConfig] | None = get_semsim_configs(
        semsim_configs_name, semsim_eval_configs
    )

    dataset_evaluation: DatasetEvaluation = DatasetEvaluation(
        dataset_id=dataset_id,
        pattern_eval_config_id=pattern_eval_config.id,
        num_samples=dataset.n_samples,
        num_positive=len(dataset_positives) if dataset_positives else None,
        num_negative=len(dataset_negatives) if dataset_negatives else None,
        semsim_configs=semsim_configs,
        sample_mod=sample_mod,
        ref_words=ref_words,
        ref_edges=ref_edges,
    )
    produce_eval_results(
        dataset_evaluation,
        pattern_eval_run,
        semsim_threshold_range,
        {lemma_match.match.edge for lemma_match in dataset.all_lemma_matches},
        dataset_positives,
        dataset_negatives,
        pattern_eval_config.hypergraph,
    )

    produce_lemma_eval_results(dataset_evaluation, dataset.lemma_matches, only_count_matches)

    dataset_evaluation_file_path: Path = get_dataset_evaluation_file_path(
        dataset.id, pattern_eval_config.id, semsim_configs_name, n_ref_edges, sample_mod,
    )

    if only_count_matches:
        remove_all_matches(dataset_evaluation)

    save_json(dataset_evaluation, dataset_evaluation_file_path)
    return dataset_evaluation

def remove_all_matches(dataset_evaluation: DatasetEvaluation):
    if dataset_evaluation.symbolic_eval_result:
        dataset_evaluation.symbolic_eval_result.matches = None
        for lemma in dataset_evaluation.lemma_symbolic_eval_results:
            dataset_evaluation.lemma_symbolic_eval_results[lemma].matches = None

    if dataset_evaluation.semsim_eval_results:
        for threshold in dataset_evaluation.semsim_eval_results:
            dataset_evaluation.semsim_eval_results[threshold].matches = None
        for lemma in dataset_evaluation.lemma_semsim_eval_results:
            for threshold in dataset_evaluation.lemma_semsim_eval_results[lemma]:
                dataset_evaluation.lemma_semsim_eval_results[lemma][threshold].matches = None


def get_positives_and_negatives(matches: list[LemmaMatch]) -> tuple[list[Hyperedge], list[Hyperedge]]:
    positives: list[Hyperedge] = [
        lemma_match.match.edge for lemma_match in matches
        if lemma_match.label == DATA_LABELS['positive']
    ]
    negatives: list[Hyperedge] = [
        lemma_match.match.edge for lemma_match in matches
        if lemma_match.label == DATA_LABELS['negative']
    ]
    assert len(positives) + len(negatives) == len(matches), (
        f"Number of positive and negative samples does not match total number of samples"
    )
    return positives, negatives


def log_dataset_statistics(dataset: LemmaDataset, positives: list[Hyperedge], negatives: list[Hyperedge]):
    num_positives: int = len(positives)
    num_negatives: int = len(negatives)

    logger.info(
        f"Dataset statistics for '{dataset.id}':\n"
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
        f"does not match hypergraph of dataset '{dataset.id}'"
    )

    # Try to get the evaluation run file if override is not enabled
    logger.info(f"### Override pattern eval run: {override}")
    pattern_eval_run: PatternEvaluationRun | None = None
    if not override:
        pattern_eval_run: PatternEvaluationRun | None = get_pattern_eval_run(
            pattern_config_id=pattern_eval_config.id, dataset_name=dataset.id
        )

    # Run the pattern evaluation if not loaded from file
    if not pattern_eval_run:
        pattern_eval_runs: list[PatternEvaluationRun] = evaluate_pattern(
            pattern_eval_config, dataset=dataset, override=override
        )
        assert len(pattern_eval_runs) == 1, "Only one eval run should be returned"
        pattern_eval_run: PatternEvaluationRun = pattern_eval_runs[0]

    return pattern_eval_config, pattern_eval_run


def get_ref_edges(
        dataset_positives: list[Hyperedge],
        n_ref_edges: int = None,
        sample_mod: int = None
) -> list[Hyperedge] | None:
    if not n_ref_edges:
        return None
    logger.info(f"Sampling {n_ref_edges} ref. edges with RNG offset '{sample_mod}'")
    if sample_mod:
        random.seed(RNG_SEED + sample_mod)
    return list(random.sample(dataset_positives, k=n_ref_edges))


def filter_dataset_positives(dataset_positives: list[Hyperedge], ref_edges: list[Hyperedge]) -> list[Hyperedge]:
    if not ref_edges:
        return dataset_positives
    return [edge for edge in dataset_positives if edge not in ref_edges]


def get_semsim_configs(
        semsim_configs_name: str,
        semsim_eval_configs: dict[str, dict[SemSimType, SemSimConfig]]
) -> dict[SemSimType, SemSimConfig] | None:
    if semsim_configs_name:
        assert semsim_configs_name in semsim_eval_configs, "Invalid SemSim configs name given"
        return semsim_eval_configs[semsim_configs_name]
    return None


def produce_eval_results(
        dataset_evaluation: DatasetEvaluation,
        pattern_eval_run: PatternEvaluationRun,
        semsim_threshold_range: list[float] | None,
        all_dataset_edges: list[Hyperedge],
        dataset_positives: list[Hyperedge] | None,
        dataset_negatives: list[Hyperedge] | None,
        hg_name: str,
):
    if semsim_threshold_range:
        assert dataset_evaluation.ref_words or dataset_evaluation.ref_edges, (
            "Either reference words or reference edges must be given when using SemSim"
        )
        dataset_evaluation.semsim_eval_results = get_semsim_eval_results(
            pattern_eval_run,
            dataset_evaluation.semsim_configs,
            semsim_threshold_range,
            dataset_evaluation.ref_words,
            dataset_evaluation.ref_edges,
            hg_name,
            all_dataset_edges,
            dataset_positives,
            dataset_negatives,
        )
    else:
        dataset_evaluation.symbolic_eval_result = get_symbolic_eval_results(
            pattern_eval_run, all_dataset_edges, dataset_positives, dataset_negatives
        )


def get_symbolic_eval_results(
        pattern_eval_run: PatternEvaluationRun,
        all_dataset_edges: set[Hyperedge],
        dataset_positives: list[Hyperedge],
        dataset_negatives: list[Hyperedge],
) -> EvaluationResult:
    logger.info(f"Processing symbolic pattern matches...")
    eval_run_positives: set[Hyperedge] = {
        match.edge for match in pattern_eval_run.matches
    }
    eval_run_negatives: set[Hyperedge] = (
        all_dataset_edges - eval_run_positives
    )

    logger.info("Done!")

    return compute_eval_result(
        list(eval_run_positives),
        list(eval_run_negatives),
        dataset_positives,
        dataset_negatives,
    )


def get_semsim_eval_results(
        pattern_eval_run: PatternEvaluationRun,
        semsim_configs: dict[SemSimType, SemSimConfig],
        semsim_threshold_range: list[float],
        ref_words: list[str],
        ref_edges: list[Hyperedge],
        hg_name: str,
        all_dataset_edges: set[Hyperedge],
        dataset_positives: list[Hyperedge],
        dataset_negatives: list[Hyperedge],
) -> dict[float, EvaluationResult]:
    hg: Hypergraph = get_hgraph(hg_name)

    # Initialize the semsim matcher if configs given
    if semsim_configs:
        for matcher_type, semsim_config in semsim_configs.items():
            init_matcher(matcher_type, semsim_config)

    total_iterations: int = len(semsim_threshold_range) * len(pattern_eval_run.matches)
    logger.info(
        f"Processing SemSim pattern matches for {len(semsim_threshold_range)} thresholds "
        f"and {len(pattern_eval_run.matches)} matches ({total_iterations} iterations)...")

    eval_scores: dict[float, EvaluationResult] = {}
    with tqdm(total=total_iterations) as pbar:
        for semsim_threshold in semsim_threshold_range:
            edge_similarities: dict[Hyperedge, float] = {}

            eval_run_positives: set[Hyperedge] = get_post_semsim_match_edges(
                edge_similarities, pattern_eval_run, semsim_threshold, ref_words, ref_edges, hg, pbar
            )
            eval_run_negatives: set[Hyperedge] = all_dataset_edges - eval_run_positives

            eval_scores[semsim_threshold] = compute_eval_result(
                eval_run_positives,
                eval_run_negatives,
                dataset_positives,
                dataset_negatives,
            )

    logger.info(f"Done!")
    return eval_scores


def get_post_semsim_match_edges(
        edge_similarities: dict[Hyperedge, float],
        pattern_eval_run: PatternEvaluationRun,
        threshold: float,
        ref_words: list[str],
        ref_edges: list[Hyperedge],
        hg: Hypergraph,
        pbar: tqdm,
) -> set[Hyperedge]:
    post_semsim_match_edges: list[Hyperedge] = []

    for match in pattern_eval_run.matches:
        # if the match has no semsim instances,
        # it cannot be excluded by semsim.
        # this should not happen, but just in case
        if not match.semsim_instances:
            post_semsim_match_edges.append(match.edge)
            pbar.update()
            continue

        if match.edge not in edge_similarities:
            edge_similarities[match.edge] = match_semsim_instances(
                semsim_instances=match.semsim_instances,
                pattern=hedge(pattern_eval_run.pattern),
                edge=match.edge,
                hg=hg,
                ref_words=ref_words,
                ref_edges=ref_edges,
                return_similarities=True
            )

        if all(similarity >= threshold for similarity in edge_similarities[match.edge]) :
            post_semsim_match_edges.append(match.edge)

        pbar.update()
    return set(post_semsim_match_edges)


def produce_lemma_eval_results(
        dataset_evaluation: DatasetEvaluation,
        lemma_matches: dict[str, list[LemmaMatch]],
        only_count_matches: bool = False,
):
    if dataset_evaluation.semsim_eval_results:
        dataset_evaluation.lemma_semsim_eval_results = {
            lemma: {
                semsim_threshold: compute_eval_result(
                    *get_lemma_positives_and_negatives(
                        lemma_matches_, semsim_eval_result.matches, only_count_matches
                    )
                ) for semsim_threshold, semsim_eval_result in dataset_evaluation.semsim_eval_results.items()
            } for lemma, lemma_matches_ in lemma_matches.items()
        }

    if dataset_evaluation.symbolic_eval_result:
        dataset_evaluation.lemma_symbolic_eval_results = {
            lemma: compute_eval_result(
                *get_lemma_positives_and_negatives(
                    lemma_matches_, dataset_evaluation.symbolic_eval_result.matches, only_count_matches
                )
            ) for lemma, lemma_matches_ in lemma_matches.items()
        }


def get_lemma_positives_and_negatives(
        lemma_matches: list[LemmaMatch],
        eval_result_matches: list[PatternMatch],
        only_count_matches: bool = False,
) -> tuple[list[Hyperedge], list[Hyperedge], list[Hyperedge] | None, list[Hyperedge] | None]:
    lemma_dataset_positives, lemma_dataset_negatives = None, None
    if not only_count_matches:
        lemma_dataset_positives, lemma_dataset_negatives = get_positives_and_negatives(lemma_matches)

    lemma_eval_run_positives: list[Hyperedge] = [
        lemma_match.match.edge for lemma_match in lemma_matches if lemma_match.match.edge in eval_result_matches
    ]
    lemma_eval_run_negatives: list[Hyperedge] = [
        lemma_match.match.edge for lemma_match in lemma_matches if lemma_match.match.edge not in eval_result_matches
    ]

    return lemma_eval_run_positives, lemma_eval_run_negatives, lemma_dataset_positives, lemma_dataset_negatives


def compute_eval_result(
        eval_run_positives: list[Hyperedge] ,
        eval_run_negatives: list[Hyperedge],
        dataset_positives: list[Hyperedge] | None,
        dataset_negatives: list[Hyperedge] | None
) -> EvaluationResult:
    if dataset_positives is None or dataset_negatives is None:
        return EvaluationResult(matches=eval_run_positives, num_matches=len(eval_run_positives))

    true_positives: list[Hyperedge] = list(set(dataset_positives) & set(eval_run_positives))
    t_p: int = len(true_positives)
    t_n: int = len(set(dataset_negatives) & set(eval_run_negatives))
    f_p: int = len(set(dataset_negatives) & set(eval_run_positives))
    f_n: int = len(set(dataset_positives) & set(eval_run_negatives))

    accuracy: float = (t_p + t_n) / (t_p + t_n + f_p + f_n) if t_p + t_n + f_p + f_n > 0 else 0.0
    precision: float = t_p / (t_p + f_p) if t_p + f_p > 0 else 0.0
    recall: float = t_p / (t_p + f_n) if t_p + f_n > 0 else 0.0
    f1: float = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    mcc: float = (
        (t_p * t_n - f_p * f_n) / ((t_p + f_p) * (t_p + f_n) * (t_n + f_p) * (t_n + f_n)) ** 0.5 if (
            (t_p + f_p) * (t_p + f_n) * (t_n + f_p) * (t_n + f_n)
        ) ** 0.5 > 0 else 0.0
    )

    return EvaluationResult(
        matches=eval_run_positives,
        correct=true_positives,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        mcc=mcc
    )


def get_dataset_evaluation_file_path(
    dataset_name: str,
    pattern_eval_config_id: str,
    semsim_configs_name: str = None,
    n_ref_edges: int = None,
    sample_mod: int = None,
):
    n_ref_edges_descriptor: str | None = f"nref-{n_ref_edges}" if n_ref_edges else None
    sample_mod_descriptor: str | None = f"smod-{sample_mod}" if sample_mod else None

    dataset_evaluation_descriptor: str = f"{dataset_name}_{EVALUATION_FILE_SUFFIX}_{pattern_eval_config_id}"

    if semsim_configs_name:
        dataset_evaluation_descriptor += f"_{semsim_configs_name}"

    dataset_evaluation_file_stem: str = dataset_evaluation_descriptor
    if n_ref_edges:
        dataset_evaluation_file_stem += f"_{n_ref_edges_descriptor}"
    if n_ref_edges and sample_mod is not None:
        dataset_evaluation_file_stem += f"_{sample_mod_descriptor}"

    dataset_evaluation_dir_path: Path = DATASET_EVAL_DIR
    if n_ref_edges:
        dataset_evaluation_dir_path /= f"{dataset_evaluation_descriptor}_{n_ref_edges_descriptor}"

    return dataset_evaluation_dir_path / f"{dataset_evaluation_file_stem}.json"
