from typing import Optional

from matplotlib import pyplot as plt
import scienceplots  # noqa
from pydantic import BaseModel

from graphbrain_semsim.datasets.models import DatasetEvaluation


EVAL_METRIC_LABELS: dict[str, str] = {
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1-Score",
    "accuracy": "Accuracy",
    "mcc": "MCC"
}

PLOT_LINE_COLORS: list[str] = [
    '#ebac23',  # yellow
    '#b80058',  # lipstick
    '#008cf9',  # azure
    '#006e00',  # green
    '#00bbad',  # caribbean
    '#d163e6',  # lavender
    '#b24502',  # brown
    '#ff9287',  # coral
    '#5954d6',  # indigo
    '#00c6f8',  # turquoise
    '#878500',  # olive
    '#00a76c',  # jade
    '#bdbdbd'   # gray
]
PLOT_LINE_STYLES: dict[str, str | tuple[int, tuple[int, int]]] = {
    'precision': 'dotted',
    'recall': 'dashed',
    'f1': 'dashdot',
    'mcc': 'solid',
    'accuracy': 'solid'
}
PLOT_LINE_WEIGHTS: dict[str, dict[str, str | float]] = {
    'bold': {
        'linewidth': '3',
        'alpha': 1.0
    },
    'light': {
        'linewidth': '1.5',
        'alpha': 0.5
    },
}

plot_line_color_idx: int = 0
plot_line_color_map: dict[str, str] = {}


class DatasetEvaluationPlotInfo(BaseModel):
    dataset_eval_id: str
    dataset_eval_name: str
    dataset_evaluation: DatasetEvaluation
    plot_line_color: Optional[str] = None
    plot_line_weight: Optional[str] = None


def get_plot_line_color(dataset_eval_id: str) -> str:
    global plot_line_color_idx
    global plot_line_color_map
    if dataset_eval_id not in plot_line_color_map:
        assert plot_line_color_idx < len(PLOT_LINE_COLORS), "Not enough plot line colors"

        plot_line_color_map[dataset_eval_id] = PLOT_LINE_COLORS[plot_line_color_idx]
        plot_line_color_idx += 1
    return plot_line_color_map[dataset_eval_id]


def reset_plot_line_color_config():
    global plot_line_color_idx
    global plot_line_color_map
    plot_line_color_idx = 0
    plot_line_color_map = {}


def plot_base_config():
    plt.style.use(['science', 'grid'])

    # Increase the font size
    plt.rcParams.update({
        'text.usetex': False,       # Disable latex because it makes your life hard
        'font.size': 14,            # Set the global font size
        'xtick.labelsize': 14,      # Set the font size for the x-axis tick labels
        'ytick.labelsize': 14,      # Set the font size for the y-axis tick labels
    })

    reset_plot_line_color_config()
