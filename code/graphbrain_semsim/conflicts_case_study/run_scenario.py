import logging
from datetime import datetime
from pathlib import Path

from graphbrain.hypergraph import Hypergraph
from graphbrain_semsim import get_hgraph, RESULT_DIR
from graphbrain_semsim.conflicts_case_study.models import (
    EvaluationScenario, EvaluationRun, PatternMatch,
)
from graphbrain_semsim.conflicts_case_study.make_pattern import make_conflict_pattern
from graphbrain_semsim.utils.general import save_json

logger = logging.getLogger(__name__)


def run(
        scenario: EvaluationScenario,
        override: bool = False,
        log_matches: bool = False
):
    # get hypergraph
    hg: Hypergraph = get_hgraph(scenario.hypergraph)

    # # initialize semsim matchers
    # logger.info("---")
    # logger.info("Initializing SemSim matchers...")
    # if scenario.semsim_configs:
    #     for semsim_type, semsim_config in scenario.semsim_configs.items():
    #         init_matcher(semsim_type, semsim_config)
    #
    # # set reference edges if given and not already set
    # if not scenario.ref_edges and scenario.id in ref_edges:
    #     scenario.ref_edges = [
    #         [ref_edge.edge for ref_edge in ref_edges_] for ref_edges_ in ref_edges[scenario.id]
    #     ]

    # eval_runs: list[EvaluationRun] = get_eval_runs(scenario, hg)

    eval_runs: list[EvaluationRun] = [prepare_eval_run(scenario)]
    for eval_run in eval_runs:
        eval_run_description: str = f"eval run [{eval_run.run_idx + 1}/{len(eval_runs)}]: '{eval_run.id}'"
        results_file_path: Path = RESULT_DIR / scenario.id / f"{eval_run.id}.json"

        logger.info(f"-----")
        if results_file_path.exists() and not override:
            logger.info(f"Skipping existing {eval_run_description}")
            continue

        logger.info(f"Executing {eval_run_description}...")
        exec_eval_run(eval_run, scenario, hg, results_file_path, log_matches=log_matches)


def prepare_eval_run(
        scenario: EvaluationScenario,
        run_idx: int = 0,
) -> EvaluationRun | None:
    pattern = make_conflict_pattern(
        preds=scenario.sub_pattern_configs["preds"],
        preps=scenario.sub_pattern_configs["preps"],
        countries=scenario.sub_pattern_configs.get("countries"),
    )

    eval_run: EvaluationRun = EvaluationRun(
        case_study=scenario.case_study,
        scenario=scenario.name,
        skip_semsim=scenario.skip_semsim,
        sub_pattern_configs=scenario.sub_pattern_configs,
        run_idx=run_idx,
        pattern=pattern,
    )

    return eval_run


def exec_eval_run(
        eval_run: EvaluationRun,
        scenario: EvaluationScenario,
        hg: Hypergraph,
        results_file_path: Path,
        log_matches: bool = False
):
    logger.info(f"Pattern: {eval_run.pattern}")
    # if eval_run.ref_edges:
    #     logger.info(f"Ref edges: {eval_run.ref_edges}")

    eval_run.start_time = datetime.now()

    eval_run.matches = []
    # for edge, variables in hg.match_sequence(scenario.hg_sequence, eval_run.pattern, ref_edges=eval_run.ref_edges):
    for match in hg.match_sequence(
            scenario.hg_sequence, eval_run.pattern, skip_semsim=eval_run.skip_semsim
    ):
        semsim_instances = None
        if eval_run.skip_semsim:
            edge, variables, semsim_instances = match
        else:
            edge, variables = match

        pattern_match: PatternMatch = PatternMatch(
            edge=str(edge),
            edge_text=hg.text(edge),
            variables=[
                {var_name: str(var_value) for var_name, var_value in variables_.items()}
                for variables_ in variables
            ],
            variables_text=[
                {var_name: hg.text(var_value) for var_name, var_value in variables_.items()}
                for variables_ in variables
            ],
            semsim_instances=semsim_instances
        )
        if log_matches:
            log_pattern_match(pattern_match)

        if logger.level == logging.DEBUG:
            input()  # wait between matches

        eval_run.matches.append(pattern_match)

    eval_run.end_time = datetime.now()
    eval_run.duration = eval_run.end_time - eval_run.start_time
    save_json(eval_run, results_file_path)


def log_pattern_match(pattern_match: PatternMatch):
    logger.info("---")
    logger.info(pattern_match.edge)
    logger.info(pattern_match.edge_text)
    logger.info(pattern_match.variables)
    logger.info(pattern_match.variables_text)
    if pattern_match.semsim_instances:
        logger.info(pattern_match.semsim_instances)


# def get_eval_runs(scenario: EvaluationScenario, hg: Hypergraph) -> list[EvaluationRun]:
#     logger.info("---")
#     logger.info("Preparing evaluation runs...")
#
#     if not scenario.threshold_values and not scenario.ref_edges:
#         return [prepare_eval_run(scenario)]
#
#     threshold_combinations: list[dict[str, float]] | None = get_threshold_combinations(scenario)
#     ref_edges_idxes: list[int] | None = list(range(len(scenario.ref_edges))) if scenario.ref_edges else None
#
#     parameter_combinations: list[tuple] = list(itertools.product(
#         *(parameter for parameter in [threshold_combinations, ref_edges_idxes] if parameter)
#     ))
#
#     # eval_runs: list[EvaluationRun] = []
#     # for run_idx, parameter_combination in enumerate(parameter_combinations):
#     #     if eval_run := prepare_eval_run(scenario, hg, run_idx, *parameter_combination):
#     #         eval_runs.append(eval_run)
#     eval_runs: list[EvaluationRun] = [
#         prepare_eval_run(scenario, hg, run_idx, *parameter_combination)
#         for run_idx, parameter_combination in enumerate(parameter_combinations)
#     ]
#
#     logger.info(f"Done. Number of evaluation runs: {len(eval_runs)}")
#     return eval_runs


# def prepare_eval_run(
#         scenario: EvaluationScenario,
#         run_idx: int = 0,
#         threshold_combination: dict[str, float] = None,
#         ref_edges_idx: int = None
#
# ) -> EvaluationRun | None:
#     sub_pattern_configs: dict[str, CompositionPattern] = deepcopy(scenario.sub_pattern_configs)
#
#     # transform scenario info into sub-pattern info
#     for sub_pattern, sub_pattern_config in sub_pattern_configs.items():
#         sub_pattern_config.components = scenario.sub_pattern_words[sub_pattern]
#         if threshold_combination:
#             sub_pattern_config.threshold = threshold_combination.get(sub_pattern)
#
#     pattern = make_conflict_pattern(
#         preds=sub_pattern_configs["preds"],
#         preps=sub_pattern_configs["preps"],
#         countries=sub_pattern_configs.get("countries"),
#     )
#
#     eval_run: EvaluationRun = EvaluationRun(
#         case_study=scenario.case_study,
#         scenario=scenario.name,
#         run_idx=run_idx,
#         pattern=pattern,
#         sub_pattern_configs=sub_pattern_configs,
#         ref_edges_config=scenario.ref_edges_configs[ref_edges_idx] if ref_edges_idx is not None else None,
#         ref_edges=scenario.ref_edges[ref_edges_idx] if ref_edges_idx is not None else None
#     )
#
#     return eval_run
#
#
# def get_threshold_combinations(scenario: EvaluationScenario) -> list[dict[str, float]] | None:
#     if not scenario.threshold_values:
#         return None
#
#     # create all combinations of thresholds
#     threshold_combinations: list[dict[str, float]] = []
#     for threshold_idx in range(get_and_validate_threshold_range_length(scenario)):
#         threshold_combination: dict[str, float] = {}
#         for sub_pattern, threshold_range in scenario.threshold_values.items():
#             threshold_combination[sub_pattern] = threshold_range[threshold_idx]
#         threshold_combinations.append(threshold_combination)
#     return threshold_combinations
#
#
# def get_and_validate_threshold_range_length(scenario: EvaluationScenario) -> int:
#     # assert all threshold ranges are of equal length or of length 1
#     threshold_range_lens: list[int] = []
#     for sub_pattern, threshold_range in scenario.threshold_values.items():
#         assert threshold_range, f"Threshold ranges for sub-pattern '{sub_pattern}' is empty."
#         if len(threshold_range) > 1:
#             threshold_range_lens.append(len(threshold_range))
#     assert all_equal(threshold_range_lens), f"Threshold ranges for sub-patterns are not of equal length (or length 1)"
#
#     if threshold_range_lens:
#         return threshold_range_lens[0]
#     return 1

