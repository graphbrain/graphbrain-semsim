import json
from pathlib import Path

from pydantic import ValidationError

from graphbrain_semsim import logger
from graphbrain_semsim.case_studies.config import PATTERN_EVAL_DIR
from graphbrain_semsim.case_studies.models import PatternEvaluationRun, PatternEvaluationConfig
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
        if file_path.is_file() and file_path.suffix == ".json":
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
