from pathlib import Path

from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import scienceplots  # noqa

from graphbrain_semsim import logger, PLOT_DIR
from graphbrain_semsim.conflicts_case_study.models import EvaluationRun, EvaluationScenario
from graphbrain_semsim.conflicts_case_study.scenario_configs import EVAL_SCENARIOS, CASE_STUDY
from graphbrain_semsim.eval_tools.plotting.utils import get_eval_runs
from graphbrain_semsim.utils.general import all_equal

# plt.rcParams.update({
#     'text.usetex': True,
#     'font.family': 'serif',
#     # 'font.serif': ['CMU']
# })

plt.style.use(['science', 'grid'])

# Increase the font size
plt.rcParams.update({
    'font.size': 14,            # Set the global font size
    'xtick.labelsize': 14,      # Set the font size for the x-axis tick labels
    'ytick.labelsize': 14,      # Set the font size for the y-axis tick labels
})


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

        # single run
        if not scenario.threshold_values:
            plot_n_results_fix_threshold(axes, scenario_id)
            continue

        # multiple runs for different thresholds
        variable_threshold_sub_patterns: list[str] = [
            sub_pattern_name for sub_pattern_name, threshold_values in scenario.threshold_values.items()
            if len(threshold_values) > 1
        ]

        if len(variable_threshold_sub_patterns) > 1 and not all_equal(
                scenario.threshold_values[sub_pattern_name] for sub_pattern_name in variable_threshold_sub_patterns
        ):
            logger.error(f"Scenario '{scenario.id}': variable threshold sub patterns must have the same values")
            continue

        plot_n_results_vs_variable_threshold(
            axes, scenario_id, variable_threshold_sub_patterns[0], x_lim_data, y_lim_data, x_lim_view, y_lim_view
        )

    figure.legend(loc='upper right')
    figure.set_dpi(300)

    plot_file_path: Path = PLOT_DIR / f"{case_study}_{plot_name}.png"
    figure.savefig(plot_file_path, bbox_inches='tight')
    logger.info(f"Plot saved to '{plot_file_path}'")


def plot_n_results_vs_variable_threshold(
        axes: Axes,
        scenario_id: str,
        variable_threshold_sub_pattern: str,
        x_lim_data: tuple[float | None, float | None] = None,
        y_lim_data: tuple[float | None, float | None] = None,
        x_lim_view: tuple[float | None, float | None] = None,
        y_lim_view: tuple[float | None, float | None] = None

):
    eval_runs: list[EvaluationRun] = get_eval_runs(scenario_id)

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


def plot_n_results_fix_threshold(
        axes: Axes,
        scenario_id: str,
):
    eval_runs: list[EvaluationRun] = get_eval_runs(scenario_id)
    assert len(eval_runs) == 1, f"Scenario '{scenario_id}' must have exactly one evaluation run"

    axes.axhline(y=len(eval_runs[0].matches), label=scenario_id, linestyle='--')


# plot_n_results_vs_threshold(
#     plot_name="n_results_vs_thresholds_semsim-fix",
#     case_study=CASE_STUDY,
#     scenario_ids=[
#         "conflicts_2-1_semsim-fix_preds",
#         "conflicts_2-2_semsim-fix_preps",
#         "conflicts_2-3_semsim-fix_preds_semsim-fix_preps"]
# )

plot_n_results_vs_threshold(
    plot_name="n_results_vs_thresholds_counties_semsim-fix",
    case_study=CASE_STUDY,
    scenario_ids=[
        "conflicts_3-1_any_countries",
        "conflicts_3-2_semsim-fix_countries"
    ]
)
