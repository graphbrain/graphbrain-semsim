import sys

from graphbrain_semsim.conflicts_case_study import logger
from graphbrain_semsim.conflicts_case_study.run_scenario import run
from graphbrain_semsim.conflicts_case_study.scenario_configs import EVAL_SCENARIOS, CASE_STUDY


def main():
    override: bool = False
    if len(sys.argv) > 1:
        if sys.argv[1] == "OVERRIDE":
            override = True
    logger.info(f"### OVERRIDE: {override} ###")

    logger.info("-----")
    logger.info(f"Running evaluation scenarios for case study '{CASE_STUDY}'")
    for scenario_idx, scenario in enumerate(EVAL_SCENARIOS):
        logger.info("-----")
        logger.info(f"Running scenario [{scenario_idx + 1}/{len(EVAL_SCENARIOS)}]: '{scenario.id}'")
        run(scenario, override=override)


if __name__ == "__main__":
    main()
