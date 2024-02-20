from graphbrain_semsim.case_studies.models import PatternEvaluationRun
from graphbrain_semsim.eval_tools.result_data.pattern_eval_runs import get_pattern_eval_run

from graphbrain_semsim.case_studies.conflicts.config import CASE_STUDY

pattern_config_names: list[str] = [
    "1-1_original-pattern",
    "1-2_pred_wildcard",
    "1-3_preps_wildcard",
    "1-4_pred_wildcard_preps_wildcard"
]

pattern_eval_run_map: dict[str, PatternEvaluationRun] = {
    pattern_config_name: get_pattern_eval_run(pattern_config_id=f"{CASE_STUDY}_{pattern_config_name}")
    for pattern_config_name in pattern_config_names
}

print("Number of matches for each pattern:")
for pattern_config_name, pattern_eval_run in pattern_eval_run_map.items():
    if pattern_eval_run:
        print(f"{pattern_config_name}: {len(pattern_eval_run.matches)}")
