"""
In this file, a function is implemented that plots the distribution of lemmas in a full dataset.
By distribution, we mean the number of matches that correspond to each lemma.
"""
from pathlib import Path

import matplotlib.pyplot as plt

from graphbrain_semsim import logger, PLOT_DIR
from graphbrain_semsim.datasets.make_dataset import get_full_dataset, get_lemma_distribution
from graphbrain_semsim.datasets.models import LemmaDataset
from graphbrain_semsim.plots import plot_base_config

from graphbrain_semsim.case_studies.conflicts.config import CASE_STUDY, HG_NAME


plot_base_config()


def plot_lemma_distribution(
        case_study: str,
        scenario_name: str,
        hg_name: str,
        var_name: str,
):
    dataset: LemmaDataset = get_full_dataset(case_study, scenario_name, hg_name=hg_name, var_name=var_name)
    logger.info(f"Dataset '{dataset.id}' contains {len(dataset.lemma_matches.keys())} lemmas.")

    # Plot
    plot_name: str = f"lemma_distribution_{dataset.id}"
    logger.info(f"Making plot '{plot_name}'...")

    fig_size: tuple[int, int] = (10, 7)
    fig, ax = plt.subplots(figsize=fig_size)

    n_matches_per_lemma: list[tuple[str, int]] = get_lemma_distribution(dataset)

    ax.bar(range(len(n_matches_per_lemma)), [n for lemma, n in n_matches_per_lemma], width=1.0, color="blue")
    ax.set_xlabel("Lemmas")
    ax.set_ylabel("Number of matches")
    ax.set_title(f"Number of matches per lemma - {dataset.id}")

    plot_file_path: Path = PLOT_DIR / "lemma_distribution" / f"{plot_name}.png"
    plt.savefig(plot_file_path)
    logger.info(f"Saved plot to '{plot_file_path}'")


if __name__ == "__main__":
    plot_lemma_distribution(CASE_STUDY, "1-2_pred_wildcard", HG_NAME, "PRED")
