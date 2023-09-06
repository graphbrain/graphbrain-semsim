import argparse

from graphbrain_semsim import logger
from graphbrain_semsim.conflicts_case_study.models import RefEdge
from graphbrain_semsim.eval_tools.utils.ref_edges import get_and_save_ref_edges
from graphbrain_semsim.conflicts_case_study.run_scenario import run
from graphbrain_semsim.conflicts_case_study.scenario_configs import (
    CASE_STUDY, EVAL_SCENARIOS, REF_EDGES_FILE_PATH,
)


def main(args):
    logger.info(f"### OVERRIDE: {args.override} ###")

    logger.info("-----")
    logger.info(f"Running evaluation scenarios for case study '{CASE_STUDY}'")

    if args.scenarios:
        scenarios_to_run = [scenario for scenario in EVAL_SCENARIOS if scenario.name in args.scenarios]
        assert len(scenarios_to_run) == len(args.scenarios), "Invalid scenario IDs"
    else:
        scenarios_to_run = EVAL_SCENARIOS

    ref_edges: dict[str, list[list[RefEdge]]] = get_and_save_ref_edges(
        CASE_STUDY, REF_EDGES_FILE_PATH, EVAL_SCENARIOS, override=args.override_ref_edges
    )

    for scenario_idx, scenario in enumerate(scenarios_to_run):
        logger.info("-----")
        logger.info(f"Running scenario [{scenario_idx + 1}/{len(scenarios_to_run)}]: '{scenario.id}'")
        run(scenario, ref_edges=ref_edges, override=args.override, log_matches=args.log_matches)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation scenarios")
    parser.add_argument("--override", action="store_true", help="Enable override mode")
    parser.add_argument("--log-matches", action="store_true", help="Enable logging of pattern matches")
    parser.add_argument("--scenarios", nargs="+", help="List of scenario IDs to run, separated by space")
    parser.add_argument("--override-ref-edges", action="store_true", help="Enable override mode for saving ref edges")
    main_args = parser.parse_args()
    main(main_args)
