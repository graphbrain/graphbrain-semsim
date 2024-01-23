import itertools
from typing import Optional

import numpy as np
from pydantic import ConfigDict, BaseModel

from graphbrain.hypergraph import Hypergraph
from graphbrain.semsim import SemSimType
from graphbrain.semsim.matcher.fixed_matcher import FixedEmbeddingMatcher
from graphbrain_semsim import logger, get_hgraph
from graphbrain_semsim.conflicts_case_study.models import EvaluationScenario, EvaluationRun
from graphbrain_semsim.conflicts_case_study.scenario_configs import EVAL_SCENARIOS

from graphbrain_semsim.eval_tools.result_data import (
    get_eval_scenario, get_pattern_eval_runs, get_variable_threshold_sub_pattern
)
from graphbrain_semsim.eval_tools.utils.lemmas import get_words_and_lemmas_from_matches


class WordLemmaEmbeddingInfo(BaseModel):
    word: str
    reference: bool = False
    embedding: Optional[np.ndarray] = None
    similarity_mean: Optional[float] = None
    similarity_max: Optional[float] = None
    lemma: Optional[str] = None
    lemma_embedding: Optional[np.ndarray] = None
    lemma_similarity_mean: Optional[float] = None
    lemma_similarity_max: Optional[float] = None

    @property
    def is_lemma(self):
        return self.word == self.lemma
    model_config = ConfigDict(arbitrary_types_allowed=True)


def filter_embeddings(
        embedding_infos: list[WordLemmaEmbeddingInfo],
        embeddings_tsne: np.ndarray,
        reference_measure: str,
        sim_range: tuple[float, float] = (0.0, 1.0)
) -> tuple[list[WordLemmaEmbeddingInfo], np.ndarray, list[float]]:
    sim_attribute: str = f"similarity_{reference_measure}"

    embedding_infos_filtered = [
        info for info in embedding_infos
        if getattr(info, sim_attribute) is None or sim_range[0] <= getattr(info, sim_attribute) <= sim_range[1]
    ]
    embeddings_tsne_filtered = embeddings_tsne[
        [
            getattr(info, sim_attribute) is None or sim_range[0] <= getattr(info, sim_attribute) <= sim_range[1]
            for info in embedding_infos
        ]
    ]
    similarities = [
        getattr(info, sim_attribute) if getattr(info, sim_attribute) is not None else 0
        for info in embedding_infos_filtered
    ]

    assert len(embedding_infos_filtered) == len(embeddings_tsne_filtered) == len(similarities), (
        f"Number of data points mismatch after filtering: "
        f"{len(embedding_infos_filtered)} != {len(embeddings_tsne_filtered)} != {len(similarities)}"
    )

    # Normalize your similarities for color mapping
    # v_min, v_max = min(similarities), max(similarities)
    # if v_min != v_max:
    #     similarities = [(sim - v_min) / (v_max - v_min) for sim in similarities]

    return embedding_infos_filtered, embeddings_tsne_filtered, similarities


def get_ref_words(eval_run: EvaluationRun, variable_name: str, scenario) -> list[str]:
    variable_threshold_sub_pattern: str = get_variable_threshold_sub_pattern(scenario)
    assert variable_threshold_sub_pattern.upper()[:-1] == variable_name, (
        f"Variable id '{variable_name}' does not match "
        f"variable threshold sub pattern '{variable_threshold_sub_pattern}'"
    )
    return eval_run.sub_pattern_configs[variable_threshold_sub_pattern].components


def get_embedding_infos(
        case_study: str,
        scenario_name: str,
        variable_name: str,
        hg_name: str,
        ref_words: list[str] = None,
) -> list[WordLemmaEmbeddingInfo]:
    logger.info("Preparing data for embedding info based plot...")
    hg: Hypergraph = get_hgraph(hg_name)

    scenario: EvaluationScenario = get_eval_scenario(
        EVAL_SCENARIOS, scenario_name=scenario_name, case_study=case_study
    )
    eval_runs: list[EvaluationRun] = get_pattern_eval_runs(scenario.id)
    # assert eval_runs and len(eval_runs) > 1, f"Scenario '{scenario.id}' has no eval runs or only one"

    fix_semsim_matcher: FixedEmbeddingMatcher = FixedEmbeddingMatcher(scenario.semsim_configs[SemSimType.FIX])
    kv_model: KeyedVectors = fix_semsim_matcher._model  # noqa

    embedding_infos: dict[str, WordLemmaEmbeddingInfo] = {}

    ref_words: list[str] = get_ref_words(eval_runs[0], variable_name, scenario) if not ref_words else ref_words
    filtered_ref_words: list[str] = fix_semsim_matcher.filter_oov(ref_words)
    assert len(filtered_ref_words) == len(ref_words), (
        f"OOV filtering removed words from reference words: {ref_words} -> {filtered_ref_words}"
    )
    for ref_word in filtered_ref_words:
        embedding_infos[ref_word] = WordLemmaEmbeddingInfo(
            word=ref_word,
            embedding=kv_model[ref_word],
            reference=True,
        )
    ref_embeddings: np.ndarray = np.vstack([info.embedding.reshape(1, -1) for info in embedding_infos.values()])

    match_words_lemmas: set[tuple[str, str]] = get_words_and_lemmas_from_matches(
        list(itertools.chain(*[eval_run.matches for eval_run in eval_runs])), variable_name, hg
    )
    # for eval_run in eval_runs:
    #     for match in eval_run.matches:
    #         match_words_lemmas.update(get_words_and_lemmas_from_match(match, variable_name, hg))

    # filtered_match_words: list[str] = fix_semsim_matcher.filter_oov([word for word, _ in match_words_lemmas])
    # filtered_match_lemmas: list[str] = fix_semsim_matcher.filter_oov([lemma for _, lemma in match_words_lemmas])
    # filtered_match_words_lemmas: list[tuple[str, str]] = [
    #     (word, lemma) for word, lemma in match_words_lemmas
    #     if word in filtered_match_words and lemma in filtered_match_lemmas
    # ]

    # for word, lemma in filtered_match_words_lemmas:
    for word, lemma in match_words_lemmas:
        if word not in embedding_infos:
            embedding: np.ndarray = kv_model[word] if word in kv_model else None
            lemma_embedding: np.ndarray = kv_model[lemma] if lemma in kv_model else None

            embedding_infos[word] = WordLemmaEmbeddingInfo(
                word=word,
                embedding=embedding,
                similarity_mean=(
                    kv_model.cosine_similarities(embedding, ref_embeddings).mean()
                    if embedding is not None else None
                ),
                similarity_max=(
                    kv_model.cosine_similarities(embedding, ref_embeddings).max(initial=0.0)
                    if embedding is not None else None
                ),
                lemma=lemma,
                lemma_embedding=lemma_embedding,
                lemma_similarity_mean=(
                    kv_model.cosine_similarities(lemma_embedding, ref_embeddings).mean()
                    if lemma_embedding is not None else None
                ),
                lemma_similarity_max=(
                    kv_model.cosine_similarities(lemma_embedding, ref_embeddings).max(initial=0.0)
                    if lemma_embedding is not None else None
                ),
            )

    logger.info(
        f"Num. of embed. infos: {len(embedding_infos)}, "
        f"Num. of ref. word embed infos: {len([info for info in embedding_infos.values() if info.reference])}, "
        f"Num. of embed infos with no word: {len([info for info in embedding_infos.values() if not info.word])}, "
        f"Num. of embed infos with lemma: {len([info for info in embedding_infos.values() if info.lemma])}, "
        f"Num. of embed infos with word=lemma: {len([info for info in embedding_infos.values() if info.is_lemma])}"
    )

    return list(embedding_infos.values())
