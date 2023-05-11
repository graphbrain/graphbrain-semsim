from pathlib import Path

from graphbrain_semsim import RESULT_DIR
from graphbrain_semsim.conflicts_case_study.models import EvaluationRun


def get_eval_runs(scenario_id: str) -> list[EvaluationRun]:
    results_dir_path: Path = RESULT_DIR / scenario_id
    eval_runs: list[EvaluationRun] = [EvaluationRun.parse_file(file_name)
                                      for file_name in results_dir_path.iterdir()]

    assert eval_runs, f"No evaluation runs found for scenario '{scenario_id}'"
    return eval_runs
