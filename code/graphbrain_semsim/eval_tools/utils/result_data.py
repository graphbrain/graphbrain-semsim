import json
from pathlib import Path

import numpy as np
from pydantic import ValidationError

from graphbrain_semsim import logger
from graphbrain_semsim.case_studies.config import PATTERN_EVAL_DIR
from graphbrain_semsim.case_studies.models import PatternEvaluationRun, PatternEvaluationConfig, PatternMatch
from graphbrain_semsim.utils.general import all_equal
from graphbrain_semsim.utils.file_handling import load_json


def get_pattern_eval_run(pattern_config_id: str, dataset_name: str = None) -> PatternEvaluationRun | None:
    # Try to get the evaluation run file
    pattern_eval_runs: list[PatternEvaluationRun] = get_pattern_eval_runs(
        pattern_config_id=pattern_config_id, dataset_name=dataset_name
    )
    if pattern_eval_runs and len(pattern_eval_runs) > 1:
        raise ValueError(f"Multiple evaluation runs found for pattern_config '{pattern_config_id}'")

    if pattern_eval_runs:
        logger.info(f"Loaded evaluation run from file for pattern config '{pattern_config_id}'")

    return pattern_eval_runs[0] if pattern_eval_runs else None


def get_pattern_eval_runs(pattern_config_id: str, dataset_name: str = None) -> list[PatternEvaluationRun] | None:
    results_dir_path: Path = PATTERN_EVAL_DIR / pattern_config_id
    if dataset_name:
        results_dir_path /= dataset_name

    if not results_dir_path.exists():
        logger.warning(f"Evaluation run directory not found: {results_dir_path}")
        return None

    eval_runs: list[PatternEvaluationRun] = []
    for file_path in results_dir_path.iterdir():
        try:
            eval_runs.append(load_json(file_path, PatternEvaluationRun))
        except (json.decoder.JSONDecodeError, ValidationError) as e:
            logger.error(f"Invalid evaluation run file: {file_path}. Error: {e}")

    if not eval_runs:
        logger.warning(f"No evaluation runs found for pattern config '{pattern_config_id}'")
    return eval_runs


def get_pattern_eval_config(
        pattern_configs: list[PatternEvaluationConfig],
        pattern_config_id: str = None,
        pattern_config_name: str = None,
        case_study: str = None
) -> PatternEvaluationConfig:
    if not pattern_config_id and not (pattern_config_name and case_study):
        raise ValueError("Either pattern_eval_config_id or pattern_eval_config_name and case_study must be provided")

    if not pattern_config_id:
        pattern_config_id: str = PatternEvaluationConfig.get_id(case_study=case_study, config_name=pattern_config_name)

    try:
        pattern_config: PatternEvaluationConfig = [s for s in pattern_configs if s.id == pattern_config_id][0]
    except IndexError:
        raise ValueError(f"Invalid pattern evaluation config id: {pattern_config_id}")

    return pattern_config


# def get_variable_threshold_sub_pattern(pattern_config: PatternEvaluationConfig) -> str | None:
#     if not pattern_config.threshold_values:
#         return None
#
#     variable_threshold_sub_patterns: list[str] = [
#         sub_pattern_name for sub_pattern_name, threshold_values in pattern_config.threshold_values.items()
#         if len(threshold_values) > 1
#     ]
#
#     if len(variable_threshold_sub_patterns) > 1 and not all_equal(
#             pattern_config.threshold_values[sub_pattern_name] for sub_pattern_name in variable_threshold_sub_patterns
#     ):
#         raise ValueError(f"Scenario '{pattern_config.id}': variable threshold sub patterns must have the same values")
#
#     if len(variable_threshold_sub_patterns) > 1:
#         logger.warning(f"Scenario '{pattern_config.id}': multiple variable threshold sub patterns found, "
#                        f"using '{variable_threshold_sub_patterns[0]}'")
#
#     return variable_threshold_sub_patterns[0]
#
#
# def get_eval_run_by_num_matches_percentile(
#         pattern_config: PatternEvaluationConfig,
#         eval_runs: list[PatternEvaluationRun],
#         percentile: int,
#         ref_edges_idx: int = None
# ) -> tuple[float, PatternEvaluationRun]:
#     if pattern_config.ref_edges_configs and len(pattern_config.ref_edges_configs) > 1 and not ref_edges_idx:
#         raise ValueError("ref_edges_idx must be provided when there are multiple ref_edges_configs")
#
#     if ref_edges_idx is not None:
#         eval_runs = [
#             eval_run for eval_run in eval_runs if eval_run.ref_edges_config == pattern_config.ref_edges_configs[ref_edges_idx]
#         ]
#
#     variable_threshold_sub_pattern: str = get_variable_threshold_sub_pattern(pattern_config)
#
#     # get the reference values for the similarity threshold
#     matches_per_threshold: list[tuple[float, list[PatternMatch], PatternEvaluationRun]] = list(sorted([
#         (
#             eval_run.sub_pattern_configs[variable_threshold_sub_pattern].threshold,
#             eval_run.matches,
#             eval_run
#         )
#         for eval_run in eval_runs
#     ], key=lambda p: p[0]))
#
#     # cut off after the first threshold with zero matches
#     matches_per_threshold_filtered: list[tuple[float, list[PatternMatch], PatternEvaluationRun]] = []
#     for threshold, matches, eval_run in matches_per_threshold:
#         matches_per_threshold_filtered.append((threshold, matches, eval_run))
#         if len(matches) == 0:
#             break
#
#     # reverse order so that searchsorted can work on ascending number of matches
#     matches_per_threshold_filtered_reversed: list[tuple[float, list[PatternMatch], PatternEvaluationRun]] = list(
#         reversed(matches_per_threshold_filtered)
#     )
#     num_matches_arr: np.ndarray = np.array([len(matches) for _, matches, _ in matches_per_threshold_filtered_reversed])
#     num_matches_at_percentile: np.ndarray = np.percentile(num_matches_arr, percentile)
#     num_matches_at_percentile_idx: int = int(np.searchsorted(num_matches_arr, num_matches_at_percentile))
#
#     threshold, matches, eval_run = matches_per_threshold_filtered_reversed[num_matches_at_percentile_idx]
#     return threshold, eval_run
