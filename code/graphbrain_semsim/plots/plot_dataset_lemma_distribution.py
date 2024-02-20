"""
In this file, a function is implemented that plots the distribution of lemmas in a full dataset.
By distribution, we mean the number of matches that correspond to each lemma.
"""
from pathlib import Path

import matplotlib.pyplot as plt
from pydantic import BaseModel

from graphbrain_semsim import logger, PLOT_DIR
from graphbrain_semsim.datasets.make_dataset import get_lemma_distribution
from graphbrain_semsim.datasets.models import LemmaDataset
from graphbrain_semsim.datasets.utils import load_dataset
from graphbrain_semsim.plots import plot_base_config, PLOT_LINE_COLORS

from graphbrain_semsim.case_studies.conflicts.config import CASE_STUDY


plot_base_config()


DATASET_PRETTY_NAMES: dict[str, str] = {
    "dataset_conflicts_1-2_pred_wildcard_full": "Filtered Base Edge Set",
    "dataset_conflicts_1-2_pred_wildcard_subsample-2000": "Conflict Dataset"
}


class DatasetConfig(BaseModel):
    case_study: str
    pattern_eval_config_name: str
    full_dataset: bool = False
    n_subsamples: int = None
    recreated: bool = False


def plot_lemma_distribution(
        dataset_configs: list[DatasetConfig],
):
    datasets: list[LemmaDataset] = [
        load_dataset(
            LemmaDataset.get_id(
                case_study=dataset_config.case_study,
                pattern_eval_config_name=dataset_config.pattern_eval_config_name,
                full_dataset=dataset_config.full_dataset,
                n_samples=dataset_config.n_subsamples,
                recreated=dataset_config.recreated,
            )
        ) for dataset_config in dataset_configs
    ]

    logger.info(f"Making plot for datasets {[dataset.id for dataset in datasets]}...")
    for dataset in datasets:
        logger.info(f"Dataset '{dataset.id}' contains {len(dataset.lemma_matches.keys())} lemmas.")

    # Plot
    plot_name: str = f"lemma_distribution_{'_'.join([dataset.id for dataset in datasets])}"
    logger.info(f"Making plot '{plot_name}'...")

    fig_size: tuple[int, int] = (10, 7)
    fig, ax = plt.subplots(figsize=fig_size)

    dataset_lemma_n_matches: dict[str, list[tuple[str, int]]] = {
        dataset.id: get_lemma_distribution(dataset) for dataset in datasets
    }

    full_dataset_lemma_n_matches: dict[str, int] = dict(dataset_lemma_n_matches[datasets[0].id])
    for dataset_idx, (dataset_id, lemma_n_matches) in enumerate(dataset_lemma_n_matches.items()):
        dataset_lemma_n_matches: dict[str, int] = dict(lemma_n_matches)
        n_matches_by_full_idx: list[int] = [
            dataset_lemma_n_matches.get(lemma, 0) for lemma in full_dataset_lemma_n_matches.keys()
        ]

        ax.bar(
            range(len(n_matches_by_full_idx)), n_matches_by_full_idx,
            width=1.0, color=PLOT_LINE_COLORS[dataset_idx],
            label=DATASET_PRETTY_NAMES[dataset_id]
        )

    ax.set_xlabel("Unique Lemma Index (sorted by number of edges in the FBES)")
    ax.set_ylabel("Number of Edges (normalized and log-scaled)")
    ax.set_yscale("log")
    # ax.set_title(f"Number of matches per lemma - {dataset.id}")
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1 ), ncol=len(datasets))

    plot_file_path: Path = PLOT_DIR / "dataset_lemma_distribution" / f"{plot_name}.png"
    plt.savefig(plot_file_path)
    logger.info(f"Saved plot to '{plot_file_path}'")


if __name__ == "__main__":
    # plot_lemma_distribution(
    #     case_study=CASE_STUDY,
    #     pattern_eval_config_name="1-2_pred_wildcard",
    #     full_dataset=True,
    # )
    # plot_lemma_distribution(
    #     case_study=CASE_STUDY,
    #     pattern_eval_config_name="1-2_pred_wildcard",
    #     n_subsamples=2000,
    #     recreated=True,
    # )

    plot_lemma_distribution(
        [
            DatasetConfig(
                case_study=CASE_STUDY,
                pattern_eval_config_name="1-2_pred_wildcard",
                full_dataset=True,
            ),
            DatasetConfig(
                case_study=CASE_STUDY,
                pattern_eval_config_name="1-2_pred_wildcard",
                n_subsamples=2000,
                recreated=True,
            )
        ]
    )
