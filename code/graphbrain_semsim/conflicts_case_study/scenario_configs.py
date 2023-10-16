import logging

from graphbrain.semsim import SemSimConfig, SemSimType
from graphbrain_semsim.conflicts_case_study.config import CASE_STUDY, HG_NAME, SEQUENCE_NAME, SUB_PATTERN_WORDS
from graphbrain_semsim.conflicts_case_study.models import (
    CompositionType, EvaluationScenario, CompositionPattern, RefEdgesConfig
)
from graphbrain_semsim.utils.general import frange

logger = logging.getLogger(__name__)


class ConflictsEvaluationScenario(EvaluationScenario):
    case_study: str = CASE_STUDY
    hypergraph: str = HG_NAME
    hg_sequence: str = SEQUENCE_NAME
    semsim_configs: dict[SemSimType, SemSimConfig] | None = {
        SemSimType.FIX: SemSimConfig(
            model_name='word2vec-google-news-300',
            similarity_threshold=0.0
        ),
        SemSimType.CTX: SemSimConfig(
            model_name='intfloat/e5-large-v2',
            similarity_threshold=0.0,
            embedding_prefix="query:"
        )
    }
    sub_pattern_words: dict[str, list[str]] = SUB_PATTERN_WORDS


EVAL_SCENARIOS: list[ConflictsEvaluationScenario] = [
    ConflictsEvaluationScenario(
        name="1_original-pattern",
        sub_pattern_configs={
            "preds": CompositionPattern(
                type=CompositionType.ANY,
            ),
            "preps": CompositionPattern(
                type=CompositionType.ANY,
            )
        }
    ),
    ConflictsEvaluationScenario(
        name="1-1_wildcard_preds",
        sub_pattern_configs={
            "preds": CompositionPattern(
                type=CompositionType.WILDCARD,
            ),
            "preps": CompositionPattern(
                type=CompositionType.ANY,
            )
        }
    ),
    ConflictsEvaluationScenario(
        name="2-1_semsim-fix_preds",
        sub_pattern_configs={
            "preds": CompositionPattern(
                type=CompositionType.SEMSIM,
                semsim_type=SemSimType.FIX,
            ),
            "preps": CompositionPattern(
                type=CompositionType.ANY,
            )
        },
        threshold_values={
            "preds": frange(0, 1, 0.01)
        }
    ),
    ConflictsEvaluationScenario(
        name="2-2_semsim-fix_preps",
        sub_pattern_configs={
            "preps": CompositionPattern(
                type=CompositionType.SEMSIM,
                semsim_type=SemSimType.FIX,
            ),
            "preds": CompositionPattern(
                type=CompositionType.ANY,
            )
        },
        threshold_values={
            "preps": frange(0, 1, 0.01)
        }
    ),
    ConflictsEvaluationScenario(
        name="2-3_semsim-fix_preds_semsim-fix_preps",
        sub_pattern_configs={
            "preds": CompositionPattern(
                type=CompositionType.SEMSIM,
                semsim_type=SemSimType.FIX,
            ),
            "preps": CompositionPattern(
                type=CompositionType.SEMSIM,
                semsim_type=SemSimType.FIX,
            )
        },
        threshold_values={
            "preds": frange(0, 1, 0.01),
            "preps": frange(0, 1, 0.01)
        }
    ),
    ConflictsEvaluationScenario(
        name="3-1_any_countries",
        sub_pattern_configs={
            "preds": CompositionPattern(
                type=CompositionType.ANY,
            ),
            "preps": CompositionPattern(
                type=CompositionType.ANY,
            ),
            "countries": CompositionPattern(
                type=CompositionType.ANY,
            )
        }
    ),
    ConflictsEvaluationScenario(
        name="3-2_semsim-fix_countries",
        sub_pattern_configs={
            "preds": CompositionPattern(
                type=CompositionType.ANY,
            ),
            "preps": CompositionPattern(
                type=CompositionType.ANY,
            ),
            "countries": CompositionPattern(
                type=CompositionType.SEMSIM,
                semsim_type=SemSimType.FIX,
            )
        },
        threshold_values={
            "countries": frange(0, 1, 0.01)
        }
    ),
    ConflictsEvaluationScenario(
        name="4-1_semsim-ctx_preds-general",
        sub_pattern_configs={
            "preds": CompositionPattern(
                type=CompositionType.SEMSIM,
                semsim_type=SemSimType.CTX,
            ),
            "preps": CompositionPattern(
                type=CompositionType.ANY,
            )
        },
        threshold_values={
            "preds": frange(0, 1, 0.1)
        },
        ref_edges_configs=[
            RefEdgesConfig(
                source_scenario="1_original-pattern",
                num_ref_edges=3,
            ),
            RefEdgesConfig(
                source_scenario="2-3_semsim-fix_preds_semsim-fix_preps",
                num_ref_edges=3,
                num_matches_percentile=50
            )
        ]
    ),
    ConflictsEvaluationScenario(
        name="4-2_semsim-ctx_preds-countries",
        sub_pattern_configs={
            "preds": CompositionPattern(
                type=CompositionType.SEMSIM,
                semsim_type=SemSimType.CTX,
            ),
            "preps": CompositionPattern(
                type=CompositionType.ANY,
            )
        },
        threshold_values={
            "preds": frange(0, 1, 0.1)
        },
        ref_edges_configs=[
            RefEdgesConfig(
                source_scenario="3-1_any_countries",
                num_ref_edges=3,
            ),
            RefEdgesConfig(
                source_scenario="3-2_semsim-fix_countries",
                num_ref_edges=3,
                num_matches_percentile=50
            )
        ]
    ),
    ConflictsEvaluationScenario(
        name="4-1_semsim-ctx_preds-general_n-refs-10",
        sub_pattern_configs={
            "preds": CompositionPattern(
                type=CompositionType.SEMSIM,
                semsim_type=SemSimType.CTX,
            ),
            "preps": CompositionPattern(
                type=CompositionType.ANY,
            )
        },
        threshold_values={
            "preds": frange(0.5, 0.6, 0.02) + frange(0.6, 0.8, 0.01)
        },
        ref_edges_configs=[
            RefEdgesConfig(
                source_scenario="1_original-pattern",
                num_ref_edges=10,
            ),
            RefEdgesConfig(
                source_scenario="2-3_semsim-fix_preds_semsim-fix_preps",
                num_ref_edges=10,
                num_matches_percentile=50
            )
        ]
    ),
    ConflictsEvaluationScenario(
        name="4-2_semsim-ctx_preds-countries_n-refs-10",
        sub_pattern_configs={
            "preds": CompositionPattern(
                type=CompositionType.SEMSIM,
                semsim_type=SemSimType.CTX,
            ),
            "preps": CompositionPattern(
                type=CompositionType.ANY,
            )
        },
        threshold_values={
            "preds": frange(0.5, 0.6, 0.02) + frange(0.6, 0.8, 0.01)
        },
        ref_edges_configs=[
            RefEdgesConfig(
                source_scenario="3-1_any_countries",
                num_ref_edges=10,
            ),
            RefEdgesConfig(
                source_scenario="3-2_semsim-fix_countries",
                num_ref_edges=10,
                num_matches_percentile=50
            )
        ]
    ),
]







