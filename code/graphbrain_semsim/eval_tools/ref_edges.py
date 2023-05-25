import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from numpy.random import default_rng

from graphbrain_semsim import logger, RNG_SEED
from graphbrain_semsim.conflicts_case_study.models import (
    EvaluationScenario, EvaluationRun, PatternMatch, RefEdge, RefEdgesConfig
)
from graphbrain_semsim.eval_tools.utils import get_eval_scenario, get_eval_runs, get_variable_threshold_sub_pattern

rng = default_rng(RNG_SEED)


def get_and_save_ref_edges(
    case_study: str,
    file_path: Path,
    scenarios: list[EvaluationScenario],
    override: bool = False,
) -> dict[str, list[list[RefEdge]]]:
    logger.info(f"Loading reference edges from '{file_path}'...")
    ref_edges: dict[str, list[list[RefEdge]]] = load_ref_edges(file_path)

    ref_edges_to_collect: list[tuple[str, RefEdgesConfig]] = list(itertools.chain(*[
        [(scenario.id, ref_edges_config) for ref_edges_config in scenario.ref_edges_configs]
        for scenario in scenarios if scenario.ref_edges_configs and (override or scenario.id not in ref_edges)
    ]))

    ref_edges_to_collect_by_source_scenario: dict[str, list[tuple[str, RefEdgesConfig]]] = {
        k: list(v) for k, v in itertools.groupby(
            sorted(ref_edges_to_collect, key=lambda t: t[1].source_scenario), key=lambda t: t[1].source_scenario
        )
    }

    ref_edges_collected: list[tuple[str, list[RefEdge]]] = list(itertools.chain(*[
        get_ref_edges_from_scenario(
            case_study=case_study,
            source_scenario=source_scenario,
            scenarios=scenarios,
            target_scenario_ref_edges_configs=target_scenario_ref_edges_configs
        ) for source_scenario, target_scenario_ref_edges_configs in ref_edges_to_collect_by_source_scenario.items()
    ]))

    for target_scenario, ref_edges_for_target in ref_edges_collected:
        if target_scenario not in ref_edges:
            ref_edges[target_scenario] = []
        ref_edges[target_scenario].append(ref_edges_for_target)

    if ref_edges_collected:
        json.dump(convert_to_dicts(ref_edges), file_path.open("w"))
        logger.info(f"Saved reference edges to '{file_path}'")

    return ref_edges


def get_ref_edges_from_scenario(
        case_study: str,
        source_scenario: str,
        scenarios: list[EvaluationScenario],
        target_scenario_ref_edges_configs: list[tuple[str, RefEdgesConfig]]
) -> list[tuple[str, list[RefEdge]]]:

    logger.info(f"Getting reference edges for case study '{case_study}' and scenario '{source_scenario}'")

    scenario: EvaluationScenario = get_eval_scenario(scenarios, scenario_name=source_scenario, case_study=case_study)
    eval_runs: list[EvaluationRun] = get_eval_runs(scenario.id)

    if len(eval_runs) == 1:
        try:
            assert (ref_edges_config.num_matches_percentile is None
                    for _, ref_edges_config in target_scenario_ref_edges_configs)
        except AssertionError:
            logger.warning(
                f"num_matches_percentile given by ref_edges_config but only one eval run found for {source_scenario=}"
            )

        return get_ref_edges_from_single_run(eval_runs[0], target_scenario_ref_edges_configs)

    # return get_ref_edges_from_multi_runs(scenario, eval_runs, target_scenario_ref_edges_configs)
    tmp = get_ref_edges_from_multi_runs(scenario, eval_runs, target_scenario_ref_edges_configs)
    return tmp


def get_ref_edges_from_multi_runs(
        scenario: EvaluationScenario,
        eval_runs: list[EvaluationRun],
        target_scenario_ref_edges_configs: list[tuple[str, RefEdgesConfig]]
) -> list[tuple[str, list[RefEdge]]]:
    variable_threshold_sub_pattern: str = get_variable_threshold_sub_pattern(scenario)

    # get the reference values for the similarity threshold
    matches_per_threshold: list[tuple[float, list[PatternMatch], str]] = list(sorted([
        (
            eval_run.sub_pattern_configs[variable_threshold_sub_pattern].threshold,
            eval_run.matches,
            eval_run.id
        )
        for eval_run in eval_runs
    ], key=lambda p: p[0]))

    # cut off after the first threshold with zero matches
    matches_per_threshold_filtered: list[tuple[float, list[PatternMatch], str]] = []
    for threshold, matches, run_id in matches_per_threshold:
        matches_per_threshold_filtered.append((threshold, matches, run_id))
        if len(matches) == 0:
            break

    # reverse order so that searchsorted can work on ascending number of matches
    matches_per_threshold_filtered_reversed: list[tuple[float, list[PatternMatch], str]] = list(
        reversed(matches_per_threshold_filtered)
    )
    num_matches: np.ndarray = np.array([len(matches) for _, matches, _ in matches_per_threshold_filtered_reversed])

    return [
        (
            target_scenario,
            get_ref_edges_by_num_matches_percentile(
                ref_edges_config=ref_edges_config,
                num_matches=num_matches,
                matches_per_threshold=matches_per_threshold_filtered_reversed)
        ) for target_scenario, ref_edges_config in target_scenario_ref_edges_configs
    ]


def get_ref_edges_by_num_matches_percentile(
        ref_edges_config: RefEdgesConfig,
        num_matches: np.ndarray[int],
        matches_per_threshold: list[tuple[float, list[PatternMatch], str]]
) -> list[RefEdge]:
    num_matches_at_percentile: np.ndarray = np.percentile(num_matches, ref_edges_config.num_matches_percentile)
    num_matches_percentile_idx: int = int(np.searchsorted(num_matches, num_matches_at_percentile))

    threshold, matches, run_id = matches_per_threshold[num_matches_percentile_idx]
    return [
        RefEdge(run_id=run_id, edge=match.edge, variable_threshold=threshold)
        for match in rng.choice(matches, size=ref_edges_config.num_ref_edges, replace=False)]


def get_ref_edges_from_single_run(
        eval_run: EvaluationRun,
        target_scenario_ref_edges_configs: list[tuple[str, RefEdgesConfig]]
) -> list[tuple[str, list[RefEdge]]]:
    return [
        (target_scenario, [RefEdge(run_id=eval_run.id, edge=match.edge)
                           for match in rng.choice(eval_run.matches, size=ref_edges_config.num_ref_edges)])
        for target_scenario, ref_edges_config in target_scenario_ref_edges_configs
    ]


def load_ref_edges(file_path: Path) -> dict[str, list[list[RefEdge]]]:
    ref_edges: dict[str, list[list[RefEdge]]] = {}

    if file_path.exists():
        try:
            ref_edges = convert_from_dicts(json.load(file_path.open()))
        except json.decoder.JSONDecodeError:
            logger.warning(f"Invalid JSON file: {file_path}")

    return ref_edges


def convert_from_dicts(ref_edges_dicts: dict[str, list[list[dict]]]) -> dict[str, list[list[RefEdge]]]:
    return {k: [[RefEdge(**v) for v in l] for l in l_l] for k, l_l in ref_edges_dicts.items()}


def convert_to_dicts(ref_edges: dict[str, list[list[RefEdge]]]) -> dict[str, list[list[dict]]]:
    return {k: [[v.dict() for v in l] for l in l_l] for k, l_l in ref_edges.items()}

