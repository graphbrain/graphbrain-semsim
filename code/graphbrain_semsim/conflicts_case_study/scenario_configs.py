import logging
from typing import Dict

from strenum import StrEnum

from graphbrain.semsim.matching.matcher import SemSimConfig, SemSimType
from graphbrain_semsim.conflicts_case_study.models import CompositionType, EvaluationScenario, CompositionPattern
from graphbrain_semsim.utils.general import frange

logger = logging.getLogger(__name__)

CASE_STUDY: str = "conflicts"
HG_NAME: str = "reddit-worldnews-01012013-01082017.hg"
SEQUENCE_NAME: str = "headers"


class ConflictsSubPattern(StrEnum):
    PREDS = "preds"
    PREPS = "preps"
    COUNTRIES = "countries"


SUB_PATTERN_WORDS: dict[ConflictsSubPattern, list[str]] = {
    ConflictsSubPattern.PREDS: ["accuse", "arrest", "clash", "condemn", "kill", "slam", "warn"],
    ConflictsSubPattern.PREPS: ["against", "for", "of", "over"],
    ConflictsSubPattern.COUNTRIES: [
        "china", "india,", "usa", "indonesia", "pakistan", "nigeria", "brazil", "bangladesh", "russia", "mexico",
        "japan", "philippines", "ethiopia", "egypt", "vietnam", "congo", "iran", "turkey", "germany", "france"
    ]
}


class ConflictsEvaluationScenario(EvaluationScenario):
    case_study: str = CASE_STUDY
    hypergraph: str = HG_NAME
    hg_sequence: str = SEQUENCE_NAME
    semsim_configs: dict[SemSimType, SemSimConfig] | None = {
        SemSimType.FIXED: SemSimConfig(
            model_name='word2vec-google-news-300',
            similarity_threshold=0.0
        ),
        SemSimType.CONTEXT: SemSimConfig(
            model_name='intfloat/e5-base',
            similarity_threshold=0.0
        )
    }
    sub_pattern_words: dict[str, list[str]] = SUB_PATTERN_WORDS


EVAL_SCENARIOS: list[ConflictsEvaluationScenario] = [
    # ConflictsEvaluationScenario(
    #     scenario="1_original-pattern",
    #     semsim_configs=None,
    #     sub_pattern_configs={
    #         "preds": CompositionPattern(
    #             type=CompositionType.ANY,
    #         ),
    #         "preps": CompositionPattern(
    #             type=CompositionType.ANY,
    #         )
    #     }
    # ),
    ConflictsEvaluationScenario(
        scenario="2-1_semsim-fix_preds",
        sub_pattern_configs={
            "preds": CompositionPattern(
                type=CompositionType.SEMSIM,
                semsim_type=SemSimType.FIXED,
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
        scenario="2-2_semsim-fix_preps",
        sub_pattern_configs={
            "preds": CompositionPattern(
                type=CompositionType.SEMSIM,
                semsim_type=SemSimType.FIXED,
            ),
            "preps": CompositionPattern(
                type=CompositionType.ANY,
            )
        },
        threshold_values={
            "preds": frange(0, 1, 0.01)
        }
    ),
    # ConflictsEvaluationScenario(
    #     scenario="2-3_semsim-fix_preds_semsim-fix_preps",
    #     pattern_config=PatternConfig(
    #         comp_types={
    #             "preds": CompositionType.SEMSIM,
    #             "preps": CompositionType.SEMSIM,
    #         },
    #         semsim_types={
    #             "preds": SemSimType.FIXED,
    #             "preps": SemSimType.FIXED,
    #         }
    #     )
    # ),
    # ConflictsEvaluationScenario(
    #     scenario="3-1_any_countries",
    #     pattern_config=PatternConfig(
    #         comp_types={
    #             "preds": CompositionType.ANY,
    #             "preps": CompositionType.ANY,
    #             "countries": CompositionType.ANY
    #         }
    #     )
    # ),
    # ConflictsEvaluationScenario(
    #     scenario="3-2_semsim-fix_countries",
    #     pattern_config=PatternConfig(
    #         comp_types={
    #             "preds": CompositionType.ANY,
    #             "preps": CompositionType.ANY,
    #             "countries": CompositionType.SEMSIM
    #         },
    #         semsim_types={
    #             "countries": SemSimType.FIXED,
    #         }
    #     ),
    #     threshold_values={
    #         "countries": frange(0, 1, 0.01)
    #     }
    # ),
    # ConflictsEvaluationScenario(
    #     scenario="4-1_semsim-ctx_preds-general",
    #     pattern_config=PatternConfig(
    #         comp_types={
    #             "preds": CompositionType.SEMSIM,
    #             "preps": CompositionType.ANY,
    #         },
    #         semsim_types={
    #             "preds": SemSimType.CONTEXT,
    #         }
    #     )
    # ),
    # ConflictsEvaluationScenario(
    #     scenario="4-2_semsim-ctx_preds-countries",
    #     pattern_config=PatternConfig(
    #         comp_types={
    #             "preds": CompositionType.SEMSIM,
    #             "preps": CompositionType.ANY,
    #         },
    #         semsim_types={
    #             "preds": SemSimType.CONTEXT,
    #         }
    #     )
    # )
]








