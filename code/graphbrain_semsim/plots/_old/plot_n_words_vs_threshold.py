from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from graphbrain_semsim import PLOT_DIR, logger
from graphbrain_semsim.conflicts_case_study.config import CASE_STUDY, HG_NAME, ConflictsSubPattern, SUB_PATTERN_WORDS

from graphbrain_semsim.plots import plot_base_config
from graphbrain_semsim.eval_tools.utils.embedding_utils import WordLemmaEmbeddingInfo, get_embedding_infos
from graphbrain_semsim.utils.file_handling import save_to_pickle, load_from_pickle

plot_base_config()

PLOT_BASE_NAME: str = "n_words_vs_thresholds"


def get_counts(
        embedding_infos: list[WordLemmaEmbeddingInfo],
        ref_sim_measure: str,
        use_lemma: bool,
        threshold_step: float = 0.01
) -> tuple[list[float], list[int]]:
    sim_attribute: str = f"{'lemma_' if use_lemma else ''}similarity_{ref_sim_measure}"
    thresholds: np.ndarray = np.arange(0, 1.01, threshold_step)

    # if only_lemma:
    #     embedding_infos = [info for info in embedding_infos if info.is_lemma]

    similarities: np.ndarray = np.array([sim for info in embedding_infos if (sim := getattr(info, sim_attribute))])
    counts: np.ndarray = np.sum(similarities[:, None] >= thresholds, axis=0)

    return list(thresholds), list(counts)


def plot(
        case_study: str,
        scenario_name: str,
        variable_name: str,
        ref_sim_measure: str,
        sim_range: tuple[float, float] = (0.0, 1.0),
        ref_words: list[str] = None,
        # only_lemma: bool = False,
        lemma_sim: bool = False
):
    util_data_dir_path: Path = PLOT_DIR / "_util_data" / f"{case_study}_{scenario_name}"

    if not (util_data_dir_path / "embedding_infos.pickle").exists():
        embedding_infos: list[WordLemmaEmbeddingInfo] = get_embedding_infos(
            case_study, scenario_name, variable_name, HG_NAME, ref_words=ref_words
        )
        save_to_pickle(embedding_infos, util_data_dir_path / "embedding_infos.pickle")

    else:
        embedding_infos: list[WordLemmaEmbeddingInfo] = load_from_pickle(util_data_dir_path / "embedding_infos.pickle")

    # counts_by_threshold: tuple[list[float], list[int]] = get_counts(
    #     embedding_infos, ref_sim_measure, only_lemma, lemma_sim
    # )
    counts_by_threshold: tuple[list[float], list[int]] = get_counts(
        embedding_infos, ref_sim_measure, lemma_sim
    )

    # Plot
    plot_name: str = f"{PLOT_BASE_NAME}_{sim_range[0]}-{sim_range[1]}_ref-sim-{ref_sim_measure}"
    if lemma_sim:
        plot_name += "_lemma-sim"
    # if only_lemma:
    #     plot_name += "_only-lemma"
    logger.info(f"Making plot '{plot_name}'...")

    fig_size: tuple[int, int] = (10, 7)
    fig, ax = plt.subplots(figsize=fig_size)

    ax.set_xlabel("Similarity threshold")
    ax.set_ylabel("Cumulative number of words")
    ax.set_title(
        # f"Number of similar {'lemmas' if only_lemma else 'words'} by similarity threshold - "
        f"Number of similar words by similarity threshold - "
        f"{ref_sim_measure} similarity {'to lemma' if lemma_sim else ''}: "
        f"{sim_range[0]}-{sim_range[1]}"
    )

    ax.scatter(*counts_by_threshold)

    save_path: Path = PLOT_DIR / PLOT_BASE_NAME / f"{plot_name}.png"
    logger.info(f"Saving plot '{save_path}'...")
    fig.savefig(save_path, bbox_inches='tight')


if __name__ == '__main__':
    # plot(
    #     case_study=CASE_STUDY,
    #     scenario_name="2-1_semsim-fix_preds",
    #     variable_name="PRED",
    #     sim_range=(0.0, 1.0),
    #     ref_sim_measure="mean"
    # )
    plot(
        case_study=CASE_STUDY,
        scenario_name="1-2_pred_wildcard",
        variable_name="PRED",
        sim_range=(0.0, 1.0),
        ref_sim_measure="max",
        ref_words=SUB_PATTERN_WORDS[ConflictsSubPattern.PREDS]
    )
    plot(
        case_study=CASE_STUDY,
        scenario_name="1-2_pred_wildcard",
        variable_name="PRED",
        sim_range=(0.0, 1.0),
        ref_sim_measure="max",
        ref_words=SUB_PATTERN_WORDS[ConflictsSubPattern.PREDS],
        lemma_sim=True
    )
    # plot(
    #     case_study=CASE_STUDY,
    #     scenario_name="1-2_pred_wildcard",
    #     variable_name="PRED",
    #     sim_range=(0.0, 1.0),
    #     ref_sim_measure="max",
    #     ref_words=SUB_PATTERN_WORDS[ConflictsSubPattern.PREDS],
    #     only_lemma=True
    # )
