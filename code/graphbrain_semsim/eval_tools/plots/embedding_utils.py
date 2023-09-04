from typing import Optional

import numpy as np
from pydantic import BaseModel

from graphbrain.semsim import SemSimType
from graphbrain.semsim.matcher.fixed_matcher import FixedEmbeddingMatcher
from graphbrain_semsim import logger
from graphbrain_semsim.conflicts_case_study.models import EvaluationScenario, EvaluationRun
from graphbrain_semsim.conflicts_case_study.scenario_configs import EVAL_SCENARIOS

from graphbrain_semsim.eval_tools.utils import get_eval_scenario, get_eval_runs, get_variable_threshold_sub_pattern


class WordEmbeddingInfo(BaseModel):
    word: str
    embedding: np.ndarray
    reference: bool = False
    similarity_mean: Optional[float] = None
    similarity_max: Optional[float] = None

    class Config:
        arbitrary_types_allowed = True


def filter_embeddings(
        embedding_infos: list[WordEmbeddingInfo],
        embeddings_tsne: np.ndarray,
        reference_measure: str,
        sim_range: tuple[float, float] = (0.0, 1.0)
) -> tuple[list[WordEmbeddingInfo], np.ndarray, list[float]]:
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


def get_embedding_infos(
        case_study: str,
        scenario_name: str,
        variable_name: str,
) -> list[WordEmbeddingInfo]:
    logger.info("Preparing data for embedding info based plot...")

    scenario: EvaluationScenario = get_eval_scenario(
        EVAL_SCENARIOS, scenario_name=scenario_name, case_study=case_study
    )
    eval_runs: list[EvaluationRun] = get_eval_runs(scenario.id)
    assert eval_runs and len(eval_runs) > 1, f"Scenario '{scenario.id}' has no eval runs or only one"

    fix_semsim_matcher: FixedEmbeddingMatcher = FixedEmbeddingMatcher(scenario.semsim_configs[SemSimType.FIX])
    kv_model: KeyedVectors = fix_semsim_matcher._model  # noqa

    variable_threshold_sub_pattern: str = get_variable_threshold_sub_pattern(scenario)
    assert variable_threshold_sub_pattern.upper()[:-1] == variable_name, (
        f"Variable name '{variable_name}' does not match "
        f"variable threshold sub pattern '{variable_threshold_sub_pattern}'"
    )

    embedding_infos: dict[str, WordEmbeddingInfo] = {}

    ref_words: list[str] = eval_runs[0].sub_pattern_configs[variable_threshold_sub_pattern].components
    filtered_ref_words: list[str] = fix_semsim_matcher.filter_oov(ref_words)
    for ref_word in filtered_ref_words:
        embedding_infos[ref_word] = WordEmbeddingInfo(
            word=ref_word,
            embedding=kv_model[ref_word],
            reference=True,
        )
    ref_embeddings: np.ndarray = np.vstack([info.embedding.reshape(1, -1) for info in embedding_infos.values()])

    match_words: set[str] = set()
    for eval_run in eval_runs:
        for match in eval_run.matches:
            match_words.update([variables_text_[variable_name] for variables_text_ in match.variables_text])

    filtered_match_words: list[str] = fix_semsim_matcher.filter_oov(list(match_words))
    for word in filtered_match_words:
        if word not in embedding_infos:
            embedding: np.ndarray = kv_model[word]
            embedding_infos[word] = WordEmbeddingInfo(
                word=word,
                embedding=embedding,
                similarity_mean=kv_model.cosine_similarities(embedding, ref_embeddings).mean(),
                similarity_max=kv_model.cosine_similarities(embedding, ref_embeddings).max(initial=0.0),
            )

    return list(embedding_infos.values())
