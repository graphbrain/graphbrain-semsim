import json
from pathlib import Path

from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import scienceplots  # noqa

from graphbrain_semsim import logger, RESULT_DIR, PLOT_DIR
from graphbrain_semsim.conflicts_case_study.models import EvaluationRun, EvaluationScenario
from graphbrain_semsim.conflicts_case_study.scenario_configs import EVAL_SCENARIOS, CASE_STUDY
from graphbrain_semsim.utils.general import all_equal

# plt.rcParams.update({
#     'text.usetex': True,
#     'font.family': 'serif',
#     # 'font.serif': ['CMU']
# })

plt.style.use(['science', 'grid'])


def plot_n_results_vs_threshold(
        plot_name: str,
        case_study: str,
        scenario_ids: list[str],
        x_lim_data: tuple[float | None, float | None] = None,
        y_lim_data: tuple[float | None, float | None] = None,
        x_lim_view: tuple[float | None, float | None] = None,
        y_lim_view: tuple[float | None, float | None] = None
):

    figure: Figure = Figure(figsize=(10, 7))
    axes: Axes = figure.add_axes((0, 0, 1, 1), xlabel="similarity threshold", ylabel="number of matches")

    for scenario_id in scenario_ids:
        try:
            scenario: EvaluationScenario = [s for s in EVAL_SCENARIOS if s.id == scenario_id][0]
        except IndexError:
            logger.error(f"Invalid scenario id: {scenario_id}")
            continue

        logger.info(f"Plotting n_results vs thresholds for scenario '{scenario_id}'...")

        variable_threshold_sub_patterns: list[str] = [
            sub_pattern_name for sub_pattern_name, threshold_values in scenario.threshold_values.items()
            if len(threshold_values) > 1
        ]

        if not variable_threshold_sub_patterns:
            logger.error(f" Scenario '{scenario_id}': must have a variable threshold sub pattern")
            continue

        if len(variable_threshold_sub_patterns) > 1 and not all_equal(
                scenario.threshold_values[sub_pattern_name] for sub_pattern_name in variable_threshold_sub_patterns
        ):
            logger.error(f"Scenario '{scenario_id}': variable threshold sub patterns must have the same values")
            continue

        variable_threshold_sub_pattern: str = variable_threshold_sub_patterns[0]

        results_dir_path: Path = RESULT_DIR / scenario_id
        eval_runs: list[EvaluationRun] = [EvaluationRun.parse_file(file_name)
                                          for file_name in results_dir_path.iterdir()]

        assert eval_runs, f"No evaluation runs found for scenario '{scenario_id}'"

        data_points: list[tuple[float, int]] = list(sorted([
            (eval_run.sub_pattern_configs[variable_threshold_sub_pattern].threshold, len(eval_run.matches))
            for eval_run in eval_runs
        ], key=lambda p: p[0]))

        if x_lim_data and x_lim_data[0]:
            data_points = list(filter(lambda p: p[0] >= x_lim_data[0], data_points))
        if x_lim_data and x_lim_data[1]:
            data_points = list(filter(lambda p: p[0] < x_lim_data[1], data_points))
        if x_lim_view:
            axes.set_xlim(left=x_lim_view[0], right=x_lim_view[1])

        if y_lim_data and y_lim_data[0]:
            data_points = list(filter(lambda p: p[1] < y_lim_data[1], data_points))
        if y_lim_data and y_lim_data[1]:
            data_points = list(filter(lambda p: p[1] < y_lim_data[1], data_points))
        if y_lim_view:
            axes.set_ylim(bottom=y_lim_view[0], top=y_lim_view[1])

        plot_data = {'threshold': [p[0] for p in data_points], 'n_results': [p[1] for p in data_points]}
        axes.plot('threshold', 'n_results', data=plot_data, label=scenario_id, marker='o', linestyle='-')

    figure.legend(loc='upper right')
    figure.set_dpi(300)

    plot_file_path: Path = PLOT_DIR / f"{case_study}_{plot_name}.png"
    figure.savefig(plot_file_path, bbox_inches='tight')
    logger.info(f"Plot saved to '{plot_file_path}'")


plot_n_results_vs_threshold(
    plot_name="n_results_vs_thresholds_semsim-fix",
    case_study=CASE_STUDY,
    scenario_ids=[
        "conflicts_2-1_semsim-fix_preds",
        "conflicts_2-2_semsim-fix_preps",
        "conflicts_2-3_semsim-fix_preds_semsim-fix_preps"]
)



# CASE_NAME_1: str = 'countries_20-most-popul_thresholds'
# CASE_NAME_2: str = 'countries_20-most-popul_thresholds-countries'
# CASE_NAME_3: str = 'countries_20-most-popul_thresholds-preds'
#
#
# plot_n_results_vs_threshold(case_name=CASE_NAME_1,
#                             y_lim_data=(None, 11000))
#
# plot_n_results_vs_threshold(case_name=CASE_NAME_1,
#                             case_name_suffix="greater-0.7",
#                             x_lim_data=(0.7, None),
#                             y_lim_view=(None, 11000))
#
# plot_n_results_vs_threshold(case_name=CASE_NAME_2,
#                             extra_info_key="countries_similarity_threshold",
#                             y_lim_view=(None, 11000))
#
# plot_n_results_vs_threshold(case_name=CASE_NAME_2,
#                             extra_info_key="countries_similarity_threshold",
#                             case_name_suffix="greater-0.4",
#                             x_lim_data=(0.4, None),
#                             y_lim_view=(None, 1600))
# #
#
# plot_n_results_vs_threshold(case_name=CASE_NAME_3, extra_info_key="preds_similarity_threshold")
