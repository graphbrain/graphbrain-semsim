import logging

from graphbrain_semsim.case_studies.conflicts.config import (
    CASE_STUDY, HG_NAME, SEQUENCE_NAME, SUB_PATTERN_WORDS, ConflictsSubPattern
)
from graphbrain_semsim.case_studies.models import (
    CompositionType, PatternEvaluationConfig, CompositionPattern
)
from graphbrain.semsim import SemSimType

logger = logging.getLogger(__name__)


class ConflictsPatternEvaluationConfig(PatternEvaluationConfig):
    case_study: str = CASE_STUDY
    hypergraph: str = HG_NAME
    hg_sequence: str = SEQUENCE_NAME
    skip_semsim: bool = False


PATTERN_CONFIGS: list[ConflictsPatternEvaluationConfig] = [
    ConflictsPatternEvaluationConfig(
        name="1-1_original-pattern",
        sub_pattern_configs={
            "pred": CompositionPattern(
                type=CompositionType.ANY,
                components=SUB_PATTERN_WORDS[ConflictsSubPattern.PREDS]
            ),
            "prep": CompositionPattern(
                type=CompositionType.ANY,
                components=SUB_PATTERN_WORDS[ConflictsSubPattern.PREPS]
            )
        }
    ),
    ConflictsPatternEvaluationConfig(
        name="1-2_pred_wildcard",
        sub_pattern_configs={
            "pred": CompositionPattern(
                type=CompositionType.WILDCARD,
            ),
            "prep": CompositionPattern(
                type=CompositionType.ANY,
                components=SUB_PATTERN_WORDS[ConflictsSubPattern.PREPS]
            )
        }
    ),
    ConflictsPatternEvaluationConfig(
        name="1-3_prep_wildcard",
        sub_pattern_configs={
            "pred": CompositionPattern(
                type=CompositionType.ANY,
                components=SUB_PATTERN_WORDS[ConflictsSubPattern.PREDS]
            ),
            "prep": CompositionPattern(
                type=CompositionType.WILDCARD,
            )
        }
    ),
    ConflictsPatternEvaluationConfig(
        name="1-4_pred_wildcard_prep_wildcard",
        sub_pattern_configs={
            "pred": CompositionPattern(
                type=CompositionType.WILDCARD,
            ),
            "prep": CompositionPattern(
                type=CompositionType.WILDCARD,
            )
        }
    ),
    ConflictsPatternEvaluationConfig(
        name="2-1_pred_semsim-fix_wildcard",
        skip_semsim=True,
        sub_pattern_configs={
            "pred": CompositionPattern(
                type=CompositionType.SEMSIM,
                semsim_type=SemSimType.FIX,
                outer_funs=["atoms"]
            ),
            "prep": CompositionPattern(
                type=CompositionType.ANY,
                components=SUB_PATTERN_WORDS[ConflictsSubPattern.PREPS]
            )
        }
    ),
    ConflictsPatternEvaluationConfig(
        name="2-2_pred_semsim-fix-lemma_wildcard",
        skip_semsim=True,
        sub_pattern_configs={
            "pred": CompositionPattern(
                type=CompositionType.SEMSIM,
                semsim_type=SemSimType.FIX,
                semsim_fix_lemma=True,
                outer_funs=["atoms"]
            ),
            "prep": CompositionPattern(
                type=CompositionType.ANY,
                components=SUB_PATTERN_WORDS[ConflictsSubPattern.PREPS]
            )
        }
    ),
    ConflictsPatternEvaluationConfig(
        name="2-3_pred_semsim-ctx_wildcard",
        skip_semsim=True,
        sub_pattern_configs={
            "pred": CompositionPattern(
                type=CompositionType.SEMSIM,
                semsim_type=SemSimType.CTX,
            ),
            "prep": CompositionPattern(
                type=CompositionType.ANY,
                components=SUB_PATTERN_WORDS[ConflictsSubPattern.PREPS]
            )
        }
    ),
]





