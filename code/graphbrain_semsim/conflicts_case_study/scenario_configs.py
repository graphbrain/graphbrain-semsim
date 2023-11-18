import logging

from graphbrain_semsim.conflicts_case_study.config import (
    CASE_STUDY, HG_NAME, SEQUENCE_NAME, SUB_PATTERN_WORDS, ConflictsSubPattern
)
from graphbrain_semsim.conflicts_case_study.models import (
    CompositionType, EvaluationScenario, CompositionPattern
)
from graphbrain.semsim import SemSimType

logger = logging.getLogger(__name__)


class ConflictsEvaluationScenario(EvaluationScenario):
    case_study: str = CASE_STUDY
    hypergraph: str = HG_NAME
    hg_sequence: str = SEQUENCE_NAME
    skip_semsim: bool = False
    # sub_pattern_words: dict[str, list[str]] = SUB_PATTERN_WORDS
    # semsim_configs: dict[SemSimType, SemSimConfig] = {
    #     SemSimType.FIX: SemSimConfig(
    #         model_name='word2vec-google-news-300',
    #         similarity_threshold=0.0
    #     ),
    #     SemSimType.CTX: SemSimConfig(
    #         model_name='intfloat/e5-large-v2',
    #         similarity_threshold=0.0,
    #         embedding_prefix="query:"
    #     )
    # }


EVAL_SCENARIOS: list[ConflictsEvaluationScenario] = [
    ConflictsEvaluationScenario(
        name="1-1_original-pattern",
        sub_pattern_configs={
            "preds": CompositionPattern(
                type=CompositionType.ANY,
                components=SUB_PATTERN_WORDS[ConflictsSubPattern.PREDS]
            ),
            "preps": CompositionPattern(
                type=CompositionType.ANY,
                components=SUB_PATTERN_WORDS[ConflictsSubPattern.PREPS]
            )
        }
    ),
    ConflictsEvaluationScenario(
        name="1-2_preds_wildcard",
        sub_pattern_configs={
            "preds": CompositionPattern(
                type=CompositionType.WILDCARD,
            ),
            "preps": CompositionPattern(
                type=CompositionType.ANY,
                components=SUB_PATTERN_WORDS[ConflictsSubPattern.PREPS]
            )
        }
    ),
    ConflictsEvaluationScenario(
        name="2-1_preds_semsim-fix_wildcard",
        skip_semsim=True,
        sub_pattern_configs={
            "preds": CompositionPattern(
                type=CompositionType.SEMSIM,
                semsim_type=SemSimType.FIX,
            ),
            "preps": CompositionPattern(
                type=CompositionType.ANY,
                components=SUB_PATTERN_WORDS[ConflictsSubPattern.PREPS]
            )
        }
    ),
    ConflictsEvaluationScenario(
        name="2-2_preds_semsim-fix_wildcard_lemma",
        skip_semsim=True,
        sub_pattern_configs={
            "preds": CompositionPattern(
                type=CompositionType.SEMSIM,
                semsim_type=SemSimType.FIX,
                semsim_fix_lemma=True
            ),
            "preps": CompositionPattern(
                type=CompositionType.ANY,
                components=SUB_PATTERN_WORDS[ConflictsSubPattern.PREPS]
            )
        }
    ),
]






