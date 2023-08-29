import json
from pathlib import Path

import numpy as np
from pydantic import ValidationError

from graphbrain_semsim import logger, RESULT_DIR
from graphbrain_semsim.conflicts_case_study.models import EvaluationRun, EvaluationScenario, PatternMatch
from graphbrain_semsim.utils.general import all_equal


def get_eval_runs(scenario_id: str) -> list[EvaluationRun]:
    results_dir_path: Path = RESULT_DIR / scenario_id
    eval_runs: list[EvaluationRun] = []
    for file_name in results_dir_path.iterdir():
        try:
            eval_runs.append(EvaluationRun.parse_file(file_name))
        except (json.decoder.JSONDecodeError, ValidationError):
            logger.error(f"Invalid evaluation run file: {file_name}")

    assert eval_runs, f"No evaluation runs found for scenario '{scenario_id}'"
    return eval_runs


def get_eval_scenario(
        scenarios: list[EvaluationScenario], scenario_id: str = None, scenario_name: str = None, case_study: str = None
) -> EvaluationScenario:
    if not scenario_id and not (scenario_name and case_study):
        raise ValueError("Either scenario_id or scenario_name and case_study must be provided")

    if not scenario_id:
        scenario_id: str = EvaluationScenario.get_id(case_study=case_study, scenario=scenario_name)

    try:
        scenario: EvaluationScenario = [s for s in scenarios if s.id == scenario_id][0]
    except IndexError:
        raise ValueError(f"Invalid scenario id: {scenario_id}")

    return scenario


def get_variable_threshold_sub_pattern(scenario: EvaluationScenario) -> str | None:
    if not scenario.threshold_values:
        return None

    variable_threshold_sub_patterns: list[str] = [
        sub_pattern_name for sub_pattern_name, threshold_values in scenario.threshold_values.items()
        if len(threshold_values) > 1
    ]

    if len(variable_threshold_sub_patterns) > 1 and not all_equal(
            scenario.threshold_values[sub_pattern_name] for sub_pattern_name in variable_threshold_sub_patterns
    ):
        raise ValueError(f"Scenario '{scenario.id}': variable threshold sub patterns must have the same values")

    if len(variable_threshold_sub_patterns) > 1:
        logger.warning(f"Scenario '{scenario.id}': multiple variable threshold sub patterns found, "
                       f"using '{variable_threshold_sub_patterns[0]}'")

    return variable_threshold_sub_patterns[0]


def get_eval_run_by_num_matches_percentile(
        scenario: EvaluationScenario,
        eval_runs: list[EvaluationRun],
        percentile: int,
        ref_edges_idx: int = None
) -> tuple[float, EvaluationRun]:
    if scenario.ref_edges_configs and len(scenario.ref_edges_configs) > 1 and not ref_edges_idx:
        raise ValueError("ref_edges_idx must be provided when there are multiple ref_edges_configs")

    if ref_edges_idx is not None:
        eval_runs = [
            eval_run for eval_run in eval_runs if eval_run.ref_edges_config == scenario.ref_edges_configs[ref_edges_idx]
        ]

    variable_threshold_sub_pattern: str = get_variable_threshold_sub_pattern(scenario)

    # get the reference values for the similarity threshold
    matches_per_threshold: list[tuple[float, list[PatternMatch], EvaluationRun]] = list(sorted([
        (
            eval_run.sub_pattern_configs[variable_threshold_sub_pattern].threshold,
            eval_run.matches,
            eval_run
        )
        for eval_run in eval_runs
    ], key=lambda p: p[0]))

    # cut off after the first threshold with zero matches
    matches_per_threshold_filtered: list[tuple[float, list[PatternMatch], EvaluationRun]] = []
    for threshold, matches, eval_run in matches_per_threshold:
        matches_per_threshold_filtered.append((threshold, matches, eval_run))
        if len(matches) == 0:
            break

    # reverse order so that searchsorted can work on ascending number of matches
    matches_per_threshold_filtered_reversed: list[tuple[float, list[PatternMatch], EvaluationRun]] = list(
        reversed(matches_per_threshold_filtered)
    )
    num_matches_arr: np.ndarray = np.array([len(matches) for _, matches, _ in matches_per_threshold_filtered_reversed])
    num_matches_at_percentile: np.ndarray = np.percentile(num_matches_arr, percentile)
    num_matches_at_percentile_idx: int = int(np.searchsorted(num_matches_arr, num_matches_at_percentile))

    threshold, matches, eval_run = matches_per_threshold_filtered_reversed[num_matches_at_percentile_idx]
    return threshold, eval_run
