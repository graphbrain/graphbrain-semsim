from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from graphbrain_semsim import PLOT_DIR, logger
from graphbrain_semsim.conflicts_case_study.scenario_configs import CASE_STUDY

from graphbrain_semsim.eval_tools.plots import plot_base_config
from graphbrain_semsim.eval_tools.plots.embedding_utils import WordEmbeddingInfo, get_embedding_infos
from graphbrain_semsim.utils.general import save_to_pickle, load_from_pickle


plot_base_config()


def get_counts(
        embedding_infos: list[WordEmbeddingInfo],
        ref_sim_measure: str,
        threshold_step: float = 0.01
) -> tuple[list[float], list[int]]:
    sim_attribute: str = f"similarity_{ref_sim_measure}"
    thresholds: np.ndarray = np.arange(0, 1.01, threshold_step)

    similarities: np.ndarray = np.array([sim for info in embedding_infos if (sim := getattr(info, sim_attribute))])
    counts: np.ndarray = np.sum(similarities[:, None] < thresholds, axis=0)

    return list(thresholds), list(counts)


def plot(
        case_study: str,
        scenario_name: str,
        variable_name: str,
        ref_sim_measure: str,
        sim_range: tuple[float, float] = (0.0, 1.0),
        annotate_words: bool = False,
):
    util_data_dir_path: Path = PLOT_DIR / "util_data" / f"{case_study}_{scenario_name}"

    if not (util_data_dir_path / "embedding_infos.pickle").exists():
        embedding_infos: list[WordEmbeddingInfo] = get_embedding_infos(
            case_study, scenario_name, variable_name
        )
        save_to_pickle(embedding_infos, util_data_dir_path / "embedding_infos.pickle")

    else:
        embedding_infos: list[WordEmbeddingInfo] = load_from_pickle(util_data_dir_path / "embedding_infos.pickle")

    counts_by_threshold: tuple[list[float], list[int]] = get_counts(embedding_infos, ref_sim_measure)

    # Plot
    plot_name: str = f"n_words_vs_threshold_{sim_range[0]}-{sim_range[1]}_ref-sim-{ref_sim_measure}"
    logger.info(f"Making plot '{plot_name}'...")

    fig_size: tuple[int, int] = (10, 7)
    fig, ax = plt.subplots(figsize=fig_size)

    ax.set_xlabel("Similarity threshold")
    ax.set_ylabel("Cumulative number of words")
    ax.set_title(f"Number of similar words by similarity threshold - "
                 f"{ref_sim_measure} similarity: {sim_range[0]}-{sim_range[1]}")

    ax.scatter(*counts_by_threshold)

    save_path: Path = PLOT_DIR / f"{plot_name}.png"
    logger.info(f"Saving plot '{save_path}'...")
    fig.savefig(save_path, bbox_inches='tight')


if __name__ == '__main__':
    plot(
        case_study=CASE_STUDY,
        scenario_name="2-1_semsim-fix_preds",
        variable_name="PRED",
        sim_range=(0.0, 1.0),
        ref_sim_measure="mean"
    )
    plot(
        case_study=CASE_STUDY,
        scenario_name="2-1_semsim-fix_preds",
        variable_name="PRED",
        sim_range=(0.0, 1.0),
        ref_sim_measure="max"
    )
