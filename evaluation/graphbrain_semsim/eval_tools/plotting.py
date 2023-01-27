import json
from pathlib import Path

import matplotlib
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from graphbrain_semsim.case_studies import get_result_dir, plot_dir


plt.rcParams.update({
    "text.usetex": True,
    # "font.family": "serif",
    # "font.sans-serif": "Computer Modern",
})


def plot_n_results_vs_threshold(
        case_name: str,
        case_name_suffix: str = "",
        extra_info_key: str = None,
        x_lim: tuple[float | None, float | None] = None,
        y_lim: tuple[float | None, float | None] = None):

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
    plot_data = {'threshold': [p[0] for p in data_points], 'n_results': [p[1] for p in data_points]}

    figure: Figure = Figure(figsize=(6, 4))
    axes: Axes = figure.add_axes((0, 0, 1, 1), xlabel="similarity threshold", ylabel="number of matches")

    if x_lim:
        axes.set_xlim(left=x_lim[0], right=x_lim[1])

    if y_lim:
        axes.set_ylim(bottom=y_lim[0], top=y_lim[1])

    axes.plot('threshold', 'n_results', data=plot_data, marker='o', linestyle='-')

    file_name = f"{case_name}_{case_name_suffix}.png" if case_name_suffix else f"{case_name}.png"
    figure.savefig(plot_dir / file_name, bbox_inches='tight')

    print(f"Plot file: {file_name}")


CASE_NAME_1: str = 'countries_20-most-popul_thresholds'
CASE_NAME_2: str = 'countries_20-most-popul_thresholds-countries'

Y_LIM = (None, 11000)

plot_n_results_vs_threshold(case_name=CASE_NAME_1,
                            y_lim=Y_LIM)

# plot_n_results_vs_threshold(case_name=CASE_NAME_1,
#                             case_name_suffix="greater-0.7",
#                             x_lim=(0.7, None),
#                             y_lim=Y_LIM)

plot_n_results_vs_threshold(case_name=CASE_NAME_2,
                            extra_info_key="countries_similarity_threshold",
                            y_lim=Y_LIM)

plot_n_results_vs_threshold(case_name=CASE_NAME_2,
                            extra_info_key="countries_similarity_threshold",
                            case_name_suffix="greater-0.4",
                            x_lim=(0.4, None),
                            y_lim=(None, 2000))



