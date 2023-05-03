import json
from pathlib import Path

from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import scienceplots  # noqa

from graphbrain_semsim.conflicts import get_result_dir, plot_dir


# plt.rcParams.update({
#     'text.usetex': True,
#     'font.family': 'serif',
#     # 'font.serif': ['CMU']
# })

plt.style.use(['science', 'grid'])


def plot_n_results_vs_threshold(
        case_name: str,
        case_name_suffix: str = "",
        extra_info_key: str = None,
        x_lim_data: tuple[float | None, float | None] = None,
        y_lim_data: tuple[float | None, float | None] = None,
        x_lim_view: tuple[float | None, float | None] = None,
        y_lim_view: tuple[float | None, float | None] = None
    ):

    results_dir_path: Path = get_result_dir(subdir=case_name)

    data_points = []
    for file_name in results_dir_path.rglob('*.json'):
        with open(results_dir_path / file_name) as fp:
            results_dict = json.load(fp)

        if extra_info_key:
            data_points.append((results_dict['extra_info'][extra_info_key], len(results_dict['results'])))
        else:
            data_points.append((results_dict['config']['similarity_threshold'], len(results_dict['results'])))

    data_points = list(sorted(data_points, key=lambda p: p[0]))

    figure: Figure = Figure(figsize=(5, 3.5))
    axes: Axes = figure.add_axes((0, 0, 1, 1), xlabel="similarity threshold", ylabel="number of matches")

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
    axes.plot('threshold', 'n_results', data=plot_data, marker='o', linestyle='-')

    file_name = f"{case_name}_{case_name_suffix}.png" if case_name_suffix else f"{case_name}.png"
    figure.savefig(plot_dir / file_name, bbox_inches='tight')

    print(f"Plot file: {file_name}")
    print(data_points)
    print("---")


# TODO: Integrate some plots

CASE_NAME_1: str = 'countries_20-most-popul_thresholds'
CASE_NAME_2: str = 'countries_20-most-popul_thresholds-countries'
CASE_NAME_3: str = 'countries_20-most-popul_thresholds-preds'


plot_n_results_vs_threshold(case_name=CASE_NAME_1,
                            y_lim_data=(None, 11000))

plot_n_results_vs_threshold(case_name=CASE_NAME_1,
                            case_name_suffix="greater-0.7",
                            x_lim_data=(0.7, None),
                            y_lim_view=(None, 11000))

plot_n_results_vs_threshold(case_name=CASE_NAME_2,
                            extra_info_key="countries_similarity_threshold",
                            y_lim_view=(None, 11000))

plot_n_results_vs_threshold(case_name=CASE_NAME_2,
                            extra_info_key="countries_similarity_threshold",
                            case_name_suffix="greater-0.4",
                            x_lim_data=(0.4, None),
                            y_lim_view=(None, 1600))
#

plot_n_results_vs_threshold(case_name=CASE_NAME_3, extra_info_key="preds_similarity_threshold")
