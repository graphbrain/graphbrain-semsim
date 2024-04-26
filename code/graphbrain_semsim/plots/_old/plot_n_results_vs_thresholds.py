from pathlib import Path
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from graphbrain_semsim import logger, PLOT_DIR
from graphbrain_semsim.conflicts_case_study.models import EvaluationRun, EvaluationScenario, RefEdgesConfig
from graphbrain_semsim.conflicts_case_study.scenario_configs import EVAL_SCENARIOS
from graphbrain_semsim.conflicts_case_study.config import CASE_STUDY
from graphbrain_semsim.eval_tools.result_data import (
    get_eval_scenario, get_pattern_eval_runs, get_variable_threshold_sub_pattern
)
from graphbrain_semsim.plots import plot_base_config


plot_base_config()


def plot_n_results_vs_threshold(
        plot_name: str,
        case_study: str,
        scenarios: list[str],
        x_lim_data: tuple[float | None, float | None] = None,
        y_lim_data: tuple[float | None, float | None] = None,
        x_lim_view: tuple[float | None, float | None] = None,
        y_lim_view: tuple[float | None, float | None] = None
):
    logger.info(f"Making plot '{plot_name}'...")

    figure: Figure = Figure(figsize=(10, 7))
    axes: Axes = figure.add_axes((0, 0, 1, 1), xlabel="similarity threshold", ylabel="number of matches")

    for scenario_name in scenarios:
        try:
            scenario: EvaluationScenario = get_eval_scenario(
                EVAL_SCENARIOS, scenario_name=scenario_name, case_study=case_study
            )
        except ValueError as e:
            logger.error(e)
            continue

        logger.info(f"Plotting n_results vs thresholds for scenario '{scenario.id}'...")

        variable_threshold_sub_pattern: str = get_variable_threshold_sub_pattern(scenario)

        # single run
        if not variable_threshold_sub_pattern:
            plot_n_results_vs_fix_threshold(axes, scenario.id)
            continue

        # multiple runs for different thresholds
        plot_n_results_vs_variable_threshold(
            axes, scenario.id, variable_threshold_sub_pattern, scenario.ref_edges_configs,
            x_lim_data, y_lim_data, x_lim_view, y_lim_view
        )

    # figure.legend(loc='upper right')
    figure.legend(loc='center left')
    figure.set_dpi(300)

    plot_file_path: Path = PLOT_DIR / f"{case_study}_{plot_name}.png"
    figure.savefig(plot_file_path, bbox_inches='tight')
    logger.info(f"Plot saved to '{plot_file_path}'")
    logger.info("-----")


def plot_n_results_vs_variable_threshold(
        axes: Axes,
        scenario_id: str,
        variable_threshold_sub_pattern: str,
        ref_edges_configs: list[RefEdgesConfig],
        x_lim_data: tuple[float | None, float | None] = None,
        y_lim_data: tuple[float | None, float | None] = None,
        x_lim_view: tuple[float | None, float | None] = None,
        y_lim_view: tuple[float | None, float | None] = None

):
    eval_runs: list[EvaluationRun] = get_pattern_eval_runs(scenario_id)

    if ref_edges_configs:
        eval_runs_by_ref_edges_source_scenario: dict[str, list[EvaluationRun]] = {
            ref_edges_config.source_scenario: [] for ref_edges_config in ref_edges_configs
        }
        for eval_run in eval_runs:
            eval_runs_by_ref_edges_source_scenario[eval_run.ref_edges_config.source_scenario].append(eval_run)

        eval_runs_: dict[str, list[EvaluationRun]] = {
            f"{scenario_id}_{ref_edges_config.source_scenario}":
                eval_runs_by_ref_edges_source_scenario[ref_edges_config.source_scenario]
            for ref_edges_config in ref_edges_configs
        }

    else:
        eval_runs_ = {scenario_id: eval_runs}

    for scenario_label, eval_runs in eval_runs_.items():
        get_and_plot_data_points(
            axes,
            eval_runs,
            scenario_label,
            variable_threshold_sub_pattern,
            x_lim_data, y_lim_data, x_lim_view, y_lim_view
        )


def get_and_plot_data_points(
        axes: Axes,
        eval_runs: list[EvaluationRun],
        scenario_label: str,
        variable_threshold_sub_pattern: str,
        x_lim_data: tuple[float | None, float | None] = None,
        y_lim_data: tuple[float | None, float | None] = None,
        x_lim_view: tuple[float | None, float | None] = None,
        y_lim_view: tuple[float | None, float | None] = None

):
    data_points: list[tuple[float, int]] = construct_and_filter_data_points(
        axes, eval_runs, variable_threshold_sub_pattern, x_lim_data, y_lim_data, x_lim_view, y_lim_view
    )

    plot_data = {'threshold': [p[0] for p in data_points], 'n_results': [p[1] for p in data_points]}
    axes.plot('threshold', 'n_results', data=plot_data, label=scenario_label, marker='o', linestyle='-')


def construct_and_filter_data_points(
        axes: Axes,
        eval_runs: list[EvaluationRun],
        variable_threshold_sub_pattern: str,
        x_lim_data: tuple[float | None, float | None] = None,
        y_lim_data: tuple[float | None, float | None] = None,
        x_lim_view: tuple[float | None, float | None] = None,
        y_lim_view: tuple[float | None, float | None] = None
) -> list[tuple[float, int]]:
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

    return data_points


def plot_n_results_vs_fix_threshold(
        axes: Axes,
        scenario_id: str,
):
    eval_runs: list[EvaluationRun] = get_pattern_eval_runs(scenario_id)
    assert len(eval_runs) == 1, f"Scenario '{scenario_id}' must have exactly one evaluation run"

    axes.axhline(y=len(eval_runs[0].matches), label=scenario_id, linestyle='--')


########################
# ====== Plotting ======
########################

# plot_n_results_vs_threshold(
#     plot_name="n_results_vs_thresholds_semsim-fix",
#     case_study=CASE_STUDY,
#     scenarios=[
#         "1_original-pattern",
#         "2-1_semsim-fix_preds",
#         "2-2_semsim-fix_preps",
#         "2-3_semsim-fix_pred_semsim-fix_preps"
#     ]
# )
#
# plot_n_results_vs_threshold(
#     plot_name="n_results_vs_thresholds_countries_semsim-fix",
#     case_study=CASE_STUDY,
#     scenarios=[
#         "3-1_any_countries",
#         "3-2_semsim-fix_countries"
#     ]
# )


# plot_n_results_vs_threshold(
#     plot_name="n_results_vs_thresholds_semsim-ctx_preds",
#     case_study=CASE_STUDY,
#     scenarios=[
#         "4-1_semsim-ctx_preds-general",
#         "4-2_semsim-ctx_preds-countries"
#     ]
# )

plot_n_results_vs_threshold(
    plot_name="n_results_vs_thresholds_semsim-ctx_preds_thresh-0.5-0.8_n-refs-10",
    case_study=CASE_STUDY,
    scenarios=[
        "4-1_semsim-ctx_preds-general_n-refs-10",
        "4-2_semsim-ctx_preds-countries_n-refs-10"
    ]
)
