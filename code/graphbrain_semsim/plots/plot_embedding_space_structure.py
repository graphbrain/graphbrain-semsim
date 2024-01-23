from pathlib import Path

import numpy as np
from matplotlib import cm
from matplotlib import colormaps
from matplotlib.pyplot import colorbar
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from sklearn.manifold import TSNE

from graphbrain_semsim import logger, PLOT_DIR
from graphbrain_semsim.eval_tools.utils.embedding_utils import filter_embeddings, get_embedding_infos, WordLemmaEmbeddingInfo
from graphbrain_semsim.utils.file_handling import save_to_pickle, load_from_pickle
from graphbrain_semsim.conflicts_case_study.config import CASE_STUDY
from graphbrain_semsim.plots import plot_base_config

plot_base_config()


REFERENCE_SIMILARITY_MEASURES = ["mean", "max"]


def plot_embedding_space_structure(
        case_study: str,
        scenario_name: str,
        variable_name: str,
        ref_sim_measure: str,
        tsne_dimensions: int,
        sim_range: tuple[float, float] = (0.0, 1.0),
        annotate_words: bool = False,
):
    assert ref_sim_measure in REFERENCE_SIMILARITY_MEASURES, f"{ref_sim_measure} not in {REFERENCE_SIMILARITY_MEASURES}"
    util_data_dir_path: Path = PLOT_DIR / "_util_data" / f"{case_study}_{scenario_name}"

    if not util_data_dir_path.exists():
        embedding_infos: list[WordLemmaEmbeddingInfo] = get_embedding_infos(
            case_study, scenario_name, variable_name
        )

        logger.info(f"Computing T-SNE for {len(embedding_infos)} word embeddings...")
        embeddings_2d: np.ndarray = TSNE(random_state=7).fit_transform(
            np.vstack([info.embedding.reshape(1, -1) for info in embedding_infos])
        )
        embeddings_1d: np.ndarray = TSNE(n_components=1, random_state=7).fit_transform(
            np.vstack([info.embedding.reshape(1, -1) for info in embedding_infos])
        )

        save_to_pickle(embedding_infos, util_data_dir_path / "embedding_infos.pickle")
        save_to_pickle(embeddings_2d, util_data_dir_path / "embeddings_2d.pickle")
        save_to_pickle(embeddings_1d, util_data_dir_path / "embeddings_1d.pickle")

    else:
        embedding_infos: list[WordLemmaEmbeddingInfo] = load_from_pickle(util_data_dir_path / "embedding_infos.pickle")
        embeddings_2d: np.ndarray = load_from_pickle(util_data_dir_path / "embeddings_2d.pickle")
        embeddings_1d: np.ndarray = load_from_pickle(util_data_dir_path / "embeddings_1d.pickle")
        assert all(obj is not None for obj in [embedding_infos, embeddings_2d, embeddings_1d]), (
            f"Could not load all necessary data from {util_data_dir_path}"
        )

    plot_name: str = f"ess_{case_study}_{scenario_name}_{tsne_dimensions}d"
    _make_ess_plot(
        embedding_infos,
        embeddings_2d,
        embeddings_1d,
        tsne_dimensions,
        ref_sim_measure,
        annotate_words,
        plot_name,
        sim_range
    )


def _make_ess_plot(
        embedding_infos: list[WordLemmaEmbeddingInfo],
        embeddings_2d: np.ndarray,
        embeddings_1d: np.ndarray,
        tsne_dimensions: int,
        ref_sim_measure: str,
        annotate_words: bool,
        plot_name: str,
        sim_range: tuple[float, float] = (0.0, 1.0)
):
    plot_name += f"_{sim_range[0]}-{sim_range[1]}_ref-sim-{ref_sim_measure}"

    if annotate_words:
        plot_name += "_annotated"

    logger.info(f"Making plot '{plot_name}'...")

    # fig_size: tuple[int, int] = (20, 14) if annotate_words else (10, 7)
    fig_size: tuple[int, int] = (10, 7)
    fig: Figure = Figure(figsize=fig_size)
    fig.set_dpi(300)

    match tsne_dimensions:
        case 1:
            _make_1d_ess_plot(
                fig, plot_name, embedding_infos, embeddings_1d, ref_sim_measure, annotate_words, sim_range
            )
        case 2:
            _make_2d_ess_plot(
                fig, plot_name, embedding_infos, embeddings_2d, ref_sim_measure, annotate_words, sim_range
            )
        case _:
            raise ValueError(f"Invalid T-SNE dimension: {tsne_dimensions}")


def _make_2d_ess_plot(
        fig: Figure,
        plot_name: str,
        embedding_infos: list[WordLemmaEmbeddingInfo],
        embeddings_2d: np.ndarray,
        ref_sim_measure: str,
        annotate_words: bool,
        sim_range: tuple[float, float] = (0.0, 1.0),
):
    ax: Axes = fig.add_axes((0, 0, 1, 1), xlabel="T-SNE embed. dim. 1", ylabel="T-SNE embed. dim. 2")
    ax.set_title(f"Embedding space structure 2D - {ref_sim_measure} similarity: {sim_range[0]}-{sim_range[1]}")

    embedding_infos, embeddings_2d, similarities = filter_embeddings(
        embedding_infos, embeddings_2d, ref_sim_measure, sim_range
    )

    # Get the color gradient based on the similarities
    cmap = colormaps['viridis']
    colors = [
        'red' if ref else cmap(sim) for ref, sim in
        zip([info.reference for info in embedding_infos], similarities)
    ]
    colorbar(cm.ScalarMappable(
        norm=Normalize(vmin=0.0, vmax=1.0), cmap=cmap), ax=ax, label='Cos. Similarity'
    )

    # Get the 2D embeddings and colors for the reference words
    ref_embeddings_2d = embeddings_2d[[info.reference for info in embedding_infos]]
    ref_colors = [color for ref, color in zip([info.reference for info in embedding_infos], colors) if ref]

    # Get the 2D embeddings and colors for the other words
    other_embeddings_2d = embeddings_2d[[not info.reference for info in embedding_infos]]
    other_colors = [color for ref, color in zip([info.reference for info in embedding_infos], colors) if not ref]

    # Plot the other words with the default shape (circle) and size
    ax.scatter(other_embeddings_2d[:, 0], other_embeddings_2d[:, 1], c=other_colors)

    # Annotate other words
    if annotate_words:
        for i, word in enumerate([info.word for info in embedding_infos if not info.reference]):
            ax.annotate(word, (other_embeddings_2d[i, 0], other_embeddings_2d[i, 1]))

    # Plot the reference words with a larger size and a different shape (e.g., square)
    ax.scatter(ref_embeddings_2d[:, 0], ref_embeddings_2d[:, 1], c=ref_colors, s=100, marker='x')

    # Annotate reference words
    for i, word in enumerate([info.word for info in embedding_infos if info.reference]):
        ax.annotate(word, (ref_embeddings_2d[i, 0], ref_embeddings_2d[i, 1]), color='red', fontsize=12,
                    textcoords="offset points", xytext=(0, 10), ha='center')

    save_path: Path = PLOT_DIR / f"{plot_name}.png"
    logger.info(f"Saving plot '{save_path}'...")
    fig.savefig(save_path, bbox_inches='tight')


def _make_1d_ess_plot(
        fig: Figure,
        plot_name: str,
        embedding_infos: list[WordLemmaEmbeddingInfo],
        embeddings_1d: np.ndarray,
        ref_sim_measure: str,
        annotate_words: bool,
        sim_range: tuple[float, float] = (0.0, 1.0),
):
    ax: Axes = fig.add_axes((0, 0, 1, 1), xlabel="T-SNE embed. dim.", ylabel="Cosine similarity")
    ax.set_title(f"Embedding space structure 1D - {ref_sim_measure} similarity: {sim_range[0]}-{sim_range[1]}")

    embedding_infos, embeddings_1d, similarities = filter_embeddings(
        embedding_infos, embeddings_1d, ref_sim_measure, sim_range
    )

    ref_mask = [info.reference for info in embedding_infos]
    other_mask = [not info.reference for info in embedding_infos]

    ref_embeddings_1d = embeddings_1d[ref_mask]
    other_embeddings_1d = embeddings_1d[other_mask]

    ref_similarities = [1.0 for sim, ref in zip(similarities, ref_mask) if ref]
    other_similarities = [sim for sim, ref in zip(similarities, ref_mask) if not ref]

    # Plot the other words with the default shape (circle) and size
    ax.scatter(other_embeddings_1d, other_similarities, c="blue")

    # Annotate other words
    if annotate_words:
        for i, word in enumerate([info.word for info in embedding_infos if not info.reference]):
            ax.annotate(word, (other_embeddings_1d[i], other_similarities[i]))

    # Plot the reference words with a larger size and a different shape
    ax.scatter(ref_embeddings_1d, ref_similarities, c="red", s=100, marker='x')

    # Annotate reference words
    for i, word in enumerate([info.word for info in embedding_infos if info.reference]):
        ax.annotate(word, (ref_embeddings_1d[i], ref_similarities[i]), color='red', fontsize=12,
                    textcoords="offset points", xytext=(0, 10), ha='center')

    # ax.invert_yaxis()

    save_path: Path = PLOT_DIR / f"{plot_name}.png"
    logger.info(f"Saving plot '{save_path}'...")
    fig.savefig(save_path, bbox_inches='tight')


if __name__ == '__main__':
    # plot_embedding_space_structure(
    #     case_study=CASE_STUDY,
    #     scenario_name="2-1_semsim-fix_preds",
    #     variable_name="PRED",
    #     sim_range=(0.0, 1.0),
    #     ref_sim_measure="mean",
    # )
    # plot_embedding_space_structure(
    #     case_study=CASE_STUDY,
    #     scenario_name="2-1_semsim-fix_preds",
    #     variable_name="PRED",
    #     sim_range=(0.0, 1.0),
    #     ref_sim_measure="max",
    # )
    # plot_embedding_space_structure(
    #     case_study=CASE_STUDY,
    #     scenario_name="2-1_semsim-fix_preds",
    #     variable_name="PRED",
    #     sim_range=(0.2, 1.0),
    #     ref_sim_measure="max"
    # )
    # plot_embedding_space_structure(
    #     case_study=CASE_STUDY,
    #     scenario_name="2-1_semsim-fix_preds",
    #     variable_name="PRED",
    #     sim_range=(0.3, 1.0),
    #     ref_sim_measure="max",
    # )
    # plot_embedding_space_structure(
    #     case_study=CASE_STUDY,
    #     scenario_name="2-1_semsim-fix_preds",
    #     variable_name="PRED",
    #     sim_range=(0.4, 1.0),
    #     ref_sim_measure="max",
    #     annotate_words=True,
    # )
    # plot_embedding_space_structure(
    #     case_study=CASE_STUDY,
    #     scenario_name="2-1_semsim-fix_preds",
    #     variable_name="PRED",
    #     sim_range=(0.5, 1.0),
    #     ref_sim_measure="max",
    #     annotate_words=True,
    # )
    # plot_embedding_space_structure(
    #     case_study=CASE_STUDY,
    #     scenario_name="2-1_semsim-fix_preds",
    #     variable_name="PRED",
    #     sim_range=(0.6, 1.0),
    #     ref_sim_measure="max",
    #     annotate_words=True,
    # )
    # plot_embedding_space_structure(
    #     case_study=CASE_STUDY,
    #     scenario_name="2-1_semsim-fix_preds",
    #     variable_name="PRED",
    #     sim_range=(0.7, 1.0),
    #     ref_sim_measure="max",
    #     annotate_words=True,
    # )
    plot_embedding_space_structure(
        case_study=CASE_STUDY,
        scenario_name="2-1_semsim-fix_preds",
        variable_name="PRED",
        tsne_dimensions=1,
        sim_range=(0.0, 1.0),
        ref_sim_measure="max",
    )
    plot_embedding_space_structure(
        case_study=CASE_STUDY,
        scenario_name="2-1_semsim-fix_preds",
        variable_name="PRED",
        tsne_dimensions=1,
        sim_range=(0.4, 1.0),
        ref_sim_measure="max",
        annotate_words=True,
    )
    plot_embedding_space_structure(
        case_study=CASE_STUDY,
        scenario_name="2-1_semsim-fix_preds",
        variable_name="PRED",
        tsne_dimensions=1,
        sim_range=(0.5, 1.0),
        ref_sim_measure="max",
        annotate_words=True,
    )
    plot_embedding_space_structure(
        case_study=CASE_STUDY,
        scenario_name="2-1_semsim-fix_preds",
        variable_name="PRED",
        tsne_dimensions=1,
        sim_range=(0.6, 1.0),
        ref_sim_measure="max",
        annotate_words=True,
    )
    plot_embedding_space_structure(
        case_study=CASE_STUDY,
        scenario_name="2-1_semsim-fix_preds",
        variable_name="PRED",
        tsne_dimensions=1,
        sim_range=(0.0, 1.0),
        ref_sim_measure="mean",
    )

