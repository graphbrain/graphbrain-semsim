import logging
from datetime import datetime
from pathlib import Path
from typing import Iterator

from tqdm import tqdm

from graphbrain.hypergraph import Hypergraph, Hyperedge
from graphbrain_semsim import get_hgraph
from graphbrain_semsim.case_studies.models import (
    PatternEvaluationConfig, PatternEvaluationRun, PatternMatch,
)
from graphbrain_semsim.case_studies.config import PATTERN_EVAL_DIR
from graphbrain_semsim.case_studies.conflicts.make_pattern import make_conflict_pattern
from graphbrain_semsim.datasets.models import LemmaDataset
from graphbrain_semsim.utils.file_handling import save_json

logger = logging.getLogger(__name__)


def evaluate_pattern(
        pattern_config: PatternEvaluationConfig,
        dataset: LemmaDataset = None,
        override: bool = False,
        log_matches: bool = False
) -> list[PatternEvaluationRun]:
    # get hypergraph
    hg: Hypergraph = get_hgraph(pattern_config.hypergraph)
    assert hg, f"Hypergraph '{pattern_config.hypergraph}' not loaded."

    # set edges subset if given
    edges_subset: list[Hyperedge] | None = [
        lemma_match.match.edge for lemma_match in dataset.all_lemma_matches
    ] if dataset else None

    # that this is a list is result of legacy implementation, where multiple runs made sense
    eval_runs: list[PatternEvaluationRun] = [prepare_eval_run(pattern_config, dataset)]
    for eval_run in eval_runs:
        eval_run_description: str = f"eval run [{eval_run.run_idx + 1}/{len(eval_runs)}]: '{eval_run.id}'"

        results_dir_path: Path = PATTERN_EVAL_DIR / pattern_config.id
        if dataset:
            results_dir_path /= dataset.id
        results_file_path: Path = results_dir_path / f"{eval_run.id}.json"

        logger.info(f"-----")
        if results_file_path.exists() and not override:
            logger.info(f"Skipping existing {eval_run_description}")
            continue

        logger.info(f"Executing {eval_run_description}...")
        exec_eval_run(eval_run, pattern_config, hg, results_file_path, edges_subset, log_matches)

    return eval_runs


def prepare_eval_run(
        pattern_config: PatternEvaluationConfig,
        dataset: LemmaDataset = None,
        run_idx: int = 0,
) -> PatternEvaluationRun | None:
    pattern = make_conflict_pattern(
        pred=pattern_config.sub_pattern_configs["pred"],
        prep=pattern_config.sub_pattern_configs["prep"],
        countries=pattern_config.sub_pattern_configs.get("countries"),
    )

    eval_run: PatternEvaluationRun = PatternEvaluationRun(
        case_study=pattern_config.case_study,
        config_name=pattern_config.name,
        skip_semsim=pattern_config.skip_semsim,
        dataset_name=dataset.id if dataset else None,
        run_idx=run_idx,
        pattern=pattern,
    )

    return eval_run


def exec_eval_run(
        eval_run: PatternEvaluationRun,
        pattern_config: PatternEvaluationConfig,
        hg: Hypergraph,
        results_file_path: Path,
        edges_subset: list[Hyperedge] = None,
        log_matches: bool = False
):
    logger.info(f"Pattern: {eval_run.pattern}")
    eval_run.start_time = datetime.now()

    eval_run.matches = []

    match_iterator: Iterator = hg.match_sequence(
        pattern_config.hg_sequence, eval_run.pattern, skip_semsim=eval_run.skip_semsim
    ) if not edges_subset else hg.match_edges(
        edges_subset, eval_run.pattern, skip_semsim=eval_run.skip_semsim
    )

    for match in tqdm(match_iterator, total=len(edges_subset) if edges_subset else None):
        semsim_instances = None
        if eval_run.skip_semsim:
            edge, variables, semsim_instances = match
        else:
            edge, variables = match

        pattern_match: PatternMatch = PatternMatch(
            edge=edge,
            edge_text=hg.text(edge),
            variables=[
                {var_name: var_value for var_name, var_value in variables_.items()}
                for variables_ in variables
            ],
            variables_text=[
                {var_name: hg.text(var_value) for var_name, var_value in variables_.items()}
                for variables_ in variables
            ],
            semsim_instances=semsim_instances
        )
        if log_matches:
            log_pattern_match(pattern_match)

        if logger.level == logging.DEBUG:
            input()  # wait between matches

        eval_run.matches.append(pattern_match)

    eval_run.end_time = datetime.now()
    eval_run.duration = eval_run.end_time - eval_run.start_time
    save_json(eval_run, results_file_path)


def log_pattern_match(pattern_match: PatternMatch):
    logger.info("---")
    logger.info(pattern_match.edge)
    logger.info(pattern_match.edge_text)
    logger.info(pattern_match.variables)
    logger.info(pattern_match.variables_text)
    if pattern_match.semsim_instances:
        logger.info(pattern_match.semsim_instances)