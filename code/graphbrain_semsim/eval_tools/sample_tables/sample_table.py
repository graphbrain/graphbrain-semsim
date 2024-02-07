# import json
# from pathlib import Path
#
# import pandas as pd
# from pydantic import BaseModel
# from numpy.random import default_rng
#
# Hyperedgephbrain.hypergraph import Hypergraph
# from graphbrain_semsim import RNG_SEED, get_hgraph
# from graphbrain_semsim.conflicts_case_study.models import EvaluationScenario, EvaluationRun
# from graphbrain_semsim.eval_tools.utils import (
#     get_eval_runs, get_eval_run_by_num_matches_percentile, get_eval_scenario, get_variable_threshold_sub_pattern
# )
# from graphbrain_semsim.conflicts_case_study.scenario_configs import EVAL_SCENARIOS
# from graphbrain_semsim.conflicts_case_study.parse_config import CASE_STUDY
#
# rng = default_rng(RNG_SEED)
#
#
# class SampleTableEntry(BaseModel):
#     scenario_name: str
#     num_samples: int
#     num_matches_percentile: int = None
#     ref_edges_idx: int = None
#
#
# OVERRIDE_DATA: bool = True
#
# TABLE_DATA_FILE_NAME: str = "table_data.json"
# LATEX_TABLE_FILE_NAME: str = "latex_table.txt"
#
#
# SAMPLE_TABLE: list[SampleTableEntry] = [
#     SampleTableEntry(
#         scenario_name="1_original-pattern",
#         num_samples=3,
#     ),
#     SampleTableEntry(
#         scenario_name="2-1_semsim-fix_preds",
#         num_samples=3,
#         num_matches_percentile=50,
#     ),
#     SampleTableEntry(
#         scenario_name="2-2_semsim-fix_preps",
#         num_samples=3,
#         num_matches_percentile=50,
#     ),
#     # SampleTableEntry(
#     #     scenario_name="3-1_any_countries",
#     #     num_samples=5,
#     #     num_matches_percentile=50,
#     # ),
#     # SampleTableEntry(
#     #     scenario_name="3-2_semsim-fix_countries",
#     #     num_samples=5,
#     #     num_matches_percentile=50,
#     # ),
# ]
#
#
# def make_sample_table(
#         case_study: str,
#         scenarios: list[EvaluationScenario],
#         table_entries: list[SampleTableEntry],
#         data_file_path: Path,
#         latex_file_path: Path,
#         override_data: bool = False,
# ):
#     if override_data or not data_file_path.exists():
#         table_data: list[dict, str] = get_sample_table_data(table_entries, case_study, scenarios)
#         data_file_path.write_text(json.dumps(table_data))
#     else:
#         table_data: list[dict, str] = json.loads(data_file_path.read_text())
#
#     df = pd.DataFrame.from_records(table_data).reset_index(drop=True)
#     latex_table: str = df.to_latex(column_format="L{4cm}L{2cm}L{5cm}L{3cm}L{5cm}L{3cm}", index=False)
#
#     print(f"\nDataframe:\n{df}\n\nLatex table:\n\n{latex_table}")
#     latex_file_path.write_text(latex_table)
#
#
# def get_sample_table_data(
#         table_entries: list[SampleTableEntry], case_study: str, scenarios: list[EvaluationScenario]
# ) -> list[dict, str]:
#     table_data: list[dict, str] = []
#     hg: Hypergraph | None = None
#
#     for table_entry in table_entries:
#         scenario = get_eval_scenario(scenarios, case_study=case_study, scenario_name=table_entry.scenario_name)
#         if not hg:
#             hg = get_hgraph(scenario.hypergraph)  # this assumes no changes to the hypergraph between scenarios
#
#         eval_runs: list[EvaluationRun] = get_eval_runs(scenario.id)
#         assert eval_runs, f"No evaluation runs found for scenario '{scenario.id}'"
#
#         variable_threshold_sub_pattern: str = get_variable_threshold_sub_pattern(scenario)
#
#         eval_run: EvaluationRun = get_eval_run_to_sample(
#             scenario, eval_runs, variable_threshold_sub_pattern, table_entry.num_matches_percentile
#         )
#
#         table_data.append({
#             "Scenario Name":
#                 escape_underscore(scenario.id) + (
#                     f"\\newline({scenario.description})" if scenario.description else ""
#                 ),
#             # "Pattern": eval_run.pattern,
#             "Pattern": "Pattern X",
#             "Samples": latex_list([
#                 match.edge_text for match in
#                 rng.choice(eval_run. matches, table_entry.num_samples, replace=False).tolist()
#             ]),
#             "Variable Threshold":
#                 f"'{variable_threshold_sub_pattern}': "
#                 f"{eval_run.sub_pattern_configs[variable_threshold_sub_pattern].threshold}\\newline "
#                 f"Percentile: {table_entry.num_matches_percentile}"
#                 if variable_threshold_sub_pattern else "-/-",
#             "Reference Edges":
#                 latex_list([hg.text(edge) for edge in eval_run.ref_edges[table_entry.ref_edges_idx]])
#                 if table_entry.ref_edges_idx is not None else "-/-",
#             "Ref. Edges Source":
#                 f"Scenario Name: {get_eval_scenario(scenarios, eval_run.ref_edges_config.source_scenario)}\\newline"
#                 f"Percentile: {eval_run.ref_edges_config.num_matches_percentile}"
#                 if table_entry.ref_edges_idx is not None else "-/-",
#         })
#
#     return table_data
#
#
# def get_eval_run_to_sample(
#         scenario: EvaluationScenario,
#         eval_runs: list[EvaluationRun],
#         variable_threshold_sub_pattern: str = None,
#         num_matches_percentile: int = None,
#         ref_edges_idx: int = None
# ) -> EvaluationRun:
#     if len(eval_runs) == 1:
#         return eval_runs[0]
#
#     assert variable_threshold_sub_pattern, "Variable threshold sub-pattern missing for multiple evaluation runs"
#     threshold, eval_run = get_eval_run_by_num_matches_percentile(
#         scenario, eval_runs, num_matches_percentile, ref_edges_idx=ref_edges_idx
#     )
#     return eval_run
#
#
# def latex_list(list_: list) -> str:
#     return f"\\newline ".join(list_)
#
#
# def escape_underscore(text: str) -> str:
#     return text.replace("_", "\\_")
#
#
# if __name__ == "__main__":
#     print(f"Generating sample table for case study '{CASE_STUDY}'...")
#     make_sample_table(
#         CASE_STUDY,
#         EVAL_SCENARIOS,
#         SAMPLE_TABLE,
#         Path(__file__).parent / TABLE_DATA_FILE_NAME,
#         Path(__file__).parent / LATEX_TABLE_FILE_NAME,
#         override_data=OVERRIDE_DATA
#     )
