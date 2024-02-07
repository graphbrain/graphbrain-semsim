import argparse

from graphbrain_semsim import logger
from graphbrain_semsim.case_studies.evaluate_pattern import evaluate_pattern
from graphbrain_semsim.case_studies.models import PatternEvaluationConfig

from graphbrain_semsim.case_studies.conflicts.config import CASE_STUDY
from graphbrain_semsim.case_studies.conflicts.pattern_configs import PATTERN_CONFIGS


def main(args):
    logger.info(f"### OVERRIDE: {args.override} ###")

    logger.info("-----")
    logger.info(f"Running pattern evaluation configs for case study '{CASE_STUDY}'")

    pattern_configs_to_run: list[PatternEvaluationConfig] = PATTERN_CONFIGS
    if args.pattern_configs:
        pattern_configs_to_run = [scenario for scenario in PATTERN_CONFIGS if scenario.name in args.pattern_configs]
        assert len(pattern_configs_to_run) == len(args.pattern_configs), "Invalid pattern parse_config IDs"

    for config_idx, config in enumerate(pattern_configs_to_run):
        logger.info("-----")
        logger.info(f"Evaluating pattern parse_config [{config_idx + 1}/{len(pattern_configs_to_run)}]: '{config.id}'")
        evaluate_pattern(config, override=args.override, log_matches=args.log_matches)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run pattern evaluation configs")
    parser.add_argument("--override", action="store_true", help="Enable override mode")
    parser.add_argument("--log-matches", action="store_true", help="Enable logging of pattern matches")
    parser.add_argument(
        "--pattern-configs", nargs="+", help="List of pattern parse_config IDs to run, separated by space"
    )
    main_args = parser.parse_args()
    main(main_args)
