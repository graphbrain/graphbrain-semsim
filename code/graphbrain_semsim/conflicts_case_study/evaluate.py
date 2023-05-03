from graphbrain_semsim.conflicts_case_study import logger
from graphbrain_semsim.conflicts_case_study.run_scenario import run
from graphbrain_semsim.conflicts_case_study.scenario_configs import EVAL_SCENARIOS, CASE_STUDY


def main():
    logger.info(f"Running evaluation scenarios for case study '{CASE_STUDY}'")
    for scenario in EVAL_SCENARIOS:
        run(scenario)


if __name__ == "__main__":
    main()
