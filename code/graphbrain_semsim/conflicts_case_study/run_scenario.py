import logging
from copy import deepcopy
from datetime import datetime

from graphbrain.hypergraph import Hypergraph
from graphbrain.semsim.interface import init_matcher
from graphbrain_semsim.conflicts_case_study import get_hgraph, RESULT_DIR
from graphbrain_semsim.conflicts_case_study.models import (
    EvaluationScenario, EvaluationRun, CompositionPattern, PatternMatch
)
from graphbrain_semsim.conflicts_case_study.make_pattern import make_conflict_pattern
from graphbrain_semsim.utils.general import all_equal, save_json

logger = logging.getLogger(__name__)


"""
parallelization
---
objects to share between processes:
- hypergraph
- semsim matchers 
--> pass the matchers through the matching functions
--> monkey-patch the matchers into the hypergraph
--> monkey-patch the _matchers in the semsim interface

"""


def run(scenario: EvaluationScenario):
    logger.info(f"Creating evaluation runs for scenario '{scenario.id}'")

    # get hypergraph
    hg: Hypergraph = get_hgraph(scenario.hypergraph)

    # initialize semsim matchers
    if scenario.semsim_configs:
        for semsim_type, semsim_config in scenario.semsim_configs.items():
            init_matcher(semsim_type, semsim_config)

    # execute evaluation runs
    for eval_run in create_eval_runs(scenario):
        logger.info("-----")
        logger.info(f"Running evaluation run {eval_run.run_idx}: '{eval_run.id}'")
        exec_eval_run(hg, scenario.hg_sequence, eval_run)
        # update scenario?


def exec_eval_run(hg: Hypergraph, sequence: str, eval_run: EvaluationRun):
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

        logger.info(pattern_match.edge)
        logger.info(pattern_match.edge_text)
        logger.info(pattern_match.variables)
        logger.info(pattern_match.variables_text)
        logger.info("---")

        eval_run.matches.append(pattern_match)

    eval_run.end_time = datetime.now()
    eval_run.duration = eval_run.end_time - eval_run.start_time

    save_json(eval_run, RESULT_DIR / eval_run.scenario / f"{eval_run.id}.json")
    logger.info("-----")


def create_eval_runs(scenario: EvaluationScenario) -> list[EvaluationRun]:
    if not scenario.threshold_values:
        return [create_eval_run(scenario)]

    threshold_combinations: list[dict[str, float]] = get_threshold_combinations(scenario)
    return [
        create_eval_run(scenario, run_idx, threshold_combination)
        for run_idx, threshold_combination in enumerate(threshold_combinations)
    ]


def create_eval_run(
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

