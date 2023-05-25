from pathlib import Path

from graphbrain_semsim import RESULT_DIR
from graphbrain_semsim.conflicts_case_study.models import EvaluationRun, EvaluationScenario
from graphbrain_semsim.utils.general import all_equal


def get_eval_runs(scenario_id: str) -> list[EvaluationRun]:
    results_dir_path: Path = RESULT_DIR / scenario_id
    eval_runs: list[EvaluationRun] = [EvaluationRun.parse_file(file_name)
                                      for file_name in results_dir_path.iterdir()]

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

    return variable_threshold_sub_patterns[0]
