import argparse

from graphbrain_semsim import logger
from graphbrain_semsim.conflicts_case_study.run_scenario import run
from graphbrain_semsim.conflicts_case_study.scenario_configs import EVAL_SCENARIOS, CASE_STUDY


def main(args):
    logger.info(f"### OVERRIDE: {args.override} ###")

    logger.info("-----")
    logger.info(f"Running evaluation scenarios for case study '{CASE_STUDY}'")

    if args.scenarios:
        # assert all(scenario_id in [scenario.scenario for scenario in EVAL_SCENARIOS] for scenario_id in args.scenarios)
        scenarios_to_run = [scenario for scenario in EVAL_SCENARIOS if scenario.scenario in args.scenarios]
        assert len(scenarios_to_run) == len(args.scenarios), "Invalid scenario IDs"
    else:
        scenarios_to_run = EVAL_SCENARIOS

    for scenario_idx, scenario in enumerate(scenarios_to_run):
        logger.info("-----")
        logger.info(f"Running scenario [{scenario_idx + 1}/{len(scenarios_to_run)}]: '{scenario.id}'")
        run(scenario, override=args.override)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation scenarios")
    parser.add_argument("--override", action="store_true", help="Enable override mode")
    parser.add_argument("--scenarios", nargs="+", help="List of scenario IDs to run, separated by space")
    main_args = parser.parse_args()
    main(main_args)
