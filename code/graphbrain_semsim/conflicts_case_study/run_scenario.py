import logging
from copy import deepcopy
from datetime import datetime
from pathlib import Path

from graphbrain.hypergraph import Hypergraph
from graphbrain.semsim.interface import init_matcher
from graphbrain_semsim import get_hgraph, RESULT_DIR
from graphbrain_semsim.conflicts_case_study.models import (
    EvaluationScenario, EvaluationRun, CompositionPattern, PatternMatch
)
from graphbrain_semsim.conflicts_case_study.make_pattern import make_conflict_pattern
from graphbrain_semsim.utils.general import all_equal, save_json

logger = logging.getLogger(__name__)


def run(scenario: EvaluationScenario, override: bool = False, log_matches: bool = False):
    # get hypergraph
    hg: Hypergraph = get_hgraph(scenario.hypergraph)

    # initialize semsim matchers
    logger.info("---")
    logger.info("Initializing SemSim matchers...")
    if scenario.semsim_configs:
        for semsim_type, semsim_config in scenario.semsim_configs.items():
            init_matcher(semsim_type, semsim_config)

    # create evaluation runs
    eval_runs: list[EvaluationRun] = make_eval_runs(scenario)
    for eval_run in eval_runs:
        # input()

        eval_run_description: str = f"Eval run [{eval_run.run_idx + 1}/{len(eval_runs)}]: '{eval_run.id}'"
        results_file_path: Path = RESULT_DIR / scenario.id / f"{eval_run.id}.json"

        logger.info(f"-----")
        if results_file_path.exists() and not override:
            logger.info(f"Skipping existing {eval_run_description}")
            continue

        logger.info(f"Executing {eval_run_description}...")
        exec_eval_run(eval_run, hg, scenario.hg_sequence, results_file_path, log_matches=log_matches)


def exec_eval_run(
        eval_run: EvaluationRun, hg: Hypergraph, sequence: str, results_file_path: Path, log_matches: bool = False
):
    # logger.info(f"Eval run info: {eval_run.json(indent=4)}")
    logger.info(f"Pattern: {eval_run.pattern}")

    eval_run.start_time = datetime.now()

    eval_run.matches = []
    for edge, variables in hg.match_sequence(sequence, eval_run.pattern):
        pattern_match: PatternMatch = PatternMatch(
            edge=str(edge),
            edge_text=hg.text(edge),
            variables=[
                {var_name: str(var_value) for var_name, var_value in variables_.items()} for variables_ in variables
            ],
            variables_text=[
                {var_name: hg.text(var_value) for var_name, var_value in variables_.items()} for variables_ in variables
            ]
        )

        if log_matches:
            log_pattern_match(pattern_match)

        eval_run.matches.append(pattern_match)

    eval_run.end_time = datetime.now()
    eval_run.duration = eval_run.end_time - eval_run.start_time

    save_json(eval_run, results_file_path)


def make_eval_runs(scenario: EvaluationScenario) -> list[EvaluationRun]:
    if not scenario.threshold_values:
        return [make_eval_run(scenario)]

    threshold_combinations: list[dict[str, float]] = get_threshold_combinations(scenario)
    return [
        make_eval_run(scenario, run_idx, threshold_combination)
        for run_idx, threshold_combination in enumerate(threshold_combinations)
    ]


def make_eval_run(
        scenario: EvaluationScenario, run_idx: int = 0, threshold_combination: dict[str, float | None] = None
):
    sub_pattern_configs: dict[str, CompositionPattern] = deepcopy(scenario.sub_pattern_configs)

    # transform scenario info into sub-pattern info
    for sub_pattern, sub_pattern_config in sub_pattern_configs.items():
        sub_pattern_config.components = scenario.sub_pattern_words[sub_pattern]
        if threshold_combination:
            sub_pattern_config.threshold = threshold_combination.get(sub_pattern)
        if scenario.ref_edges:
            sub_pattern_config.ref_edges = scenario.ref_edges.get(sub_pattern)

    pattern = make_conflict_pattern(
        preds=sub_pattern_configs["preds"],
        preps=sub_pattern_configs["preps"],
        countries=sub_pattern_configs.get("countries"),
    )

    return EvaluationRun(
        case_study=scenario.case_study,
        scenario=scenario.scenario,
        run_idx=run_idx,
        pattern=pattern,
        sub_pattern_configs=sub_pattern_configs
    )


def log_pattern_match(pattern_match: PatternMatch):
    logger.info(pattern_match.edge)
    logger.info(pattern_match.edge_text)
    logger.info(pattern_match.variables)
    logger.info(pattern_match.variables_text)
    logger.info("---")


def get_threshold_combinations(scenario: EvaluationScenario):
    # create all combinations of thresholds
    threshold_combinations: list[dict[str, float]] = []
    for threshold_idx in range(get_and_validate_threshold_range_length(scenario)):
        threshold_combination: dict[str, float] = {}
        for sub_pattern, threshold_range in scenario.threshold_values.items():
            if len(threshold_range) == 1:
                threshold_combination[sub_pattern] = threshold_range[0]
            else:
                threshold_combination[sub_pattern] = threshold_range[threshold_idx]
        threshold_combinations.append(threshold_combination)
    return threshold_combinations


def get_and_validate_threshold_range_length(scenario: EvaluationScenario) -> int:
    # assert all threshold ranges are of equal length or of length 1
    threshold_range_lens: list[int] = []
    for sub_pattern, threshold_range in scenario.threshold_values.items():
        assert threshold_range, f"Threshold ranges for sub-pattern '{sub_pattern}' is empty."
        if len(threshold_range) > 1:
            threshold_range_lens.append(len(threshold_range))
    assert all_equal(threshold_range_lens), f"Threshold ranges for sub-patterns are not of equal length (or 1)"

    if threshold_range_lens:
        return threshold_range_lens[0]
    return 1

